import { GoogleGenerativeAI } from "@google/generative-ai";
import type { AdapterOptions } from "./types.js";
import type { ImageGenerationAdapter } from "adminforth";

export default class ImageGenerationAdapterNanoBanana implements ImageGenerationAdapter {
  options: AdapterOptions;
  private genAI: GoogleGenerativeAI;

  constructor(options: AdapterOptions) {
    this.options = options;
    this.options.model = options.model || 'gemini-3.1-flash-image-preview';
    this.genAI = new GoogleGenerativeAI(this.options.nanoBananaApiKey);
  }

  validate(): void {
    if (!this.options.nanoBananaApiKey) {
      throw new Error("Nano Banana (Google) API Key is required");
    }
  }

  outputImagesMaxCountSupported(): number { return 4; }

  outputDimensionsSupported(): string[] {
    return ['1024x1024', '1440x1024', '1024x1440'];
  }

  inputFileExtensionSupported(): string[] {
    return ['png', 'jpg', 'jpeg'];
  }

  private async urlToGenerativePart(url: string) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const contentType = response.headers.get("content-type") || "image/jpeg";
    return {
      inlineData: {
        data: Buffer.from(buffer).toString("base64"),
        mimeType: contentType,
      },
    };
  }

  async generate({
    prompt,
    inputFiles,
    n = 1,
    size = '1024x1024',
    extraParams = {}
  }: {
    prompt: string;
    inputFiles: string[];
    size?: string;
    n?: number;
    extraParams?: Record<string, any>;
  }): Promise<{ imageURLs?: string[]; error?: string }> {
    try {

      const model = this.genAI.getGenerativeModel({ model: this.options.model });

      const imageParts = await Promise.all(
        inputFiles.map(url => this.urlToGenerativePart(url))
      );

      const contents = [{
        role: 'user',
        parts: [
          ...imageParts,
          { text: `Generate a new image based on the provided images: ${prompt}` }
        ]
      }];

      const generationConfig: any = { candidateCount: n };
      if (extraParams) Object.assign(generationConfig, extraParams);

      const result = await model.generateContent({
        contents,
        generationConfig
      });

      const images = result.response.candidates?.map(c => {
        const part = c.content.parts.find(p => p.inlineData);
        return part ? `data:${part.inlineData.mimeType};base64,${part.inlineData.data}` : null;
      }).filter(Boolean) as string[];

      if (images.length === 0) {
        const textResponse = result.response.text();
        return { error: `Model returned text instead of image: ${textResponse}` };
      }

      return { imageURLs: images };

    } catch (err) {
      return { error: err instanceof Error ? err.message : String(err) };
    }
  }
}