import { GoogleGenerativeAI } from "@google/generative-ai";
import type { AdapterOptions } from "./types.js";
import type { ImageGenerationAdapter } from "adminforth";

export default class ImageGenerationAdapterNanoBanana implements ImageGenerationAdapter {
  options: AdapterOptions;
  private genAI: GoogleGenerativeAI;

  constructor(options: AdapterOptions) {
    this.options = options;
    this.options.model = options.model || 'imagen-3';
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

  async generate({
    prompt,
    inputFiles,
    n = 1,
    size = '1024x1024',
  }: {
    prompt: string;
    inputFiles: string[];
    size?: string;
    n?: number;
  }): Promise<{ imageURLs?: string[]; error?: string; }> {
    try {
      const model = this.genAI.getGenerativeModel({ model: this.options.model });

      const result = await model.generateContent({
        contents: [{
          role: 'user',
          parts: [{ text: prompt }]
        }],
        generationConfig: {
          candidateCount: n,
        }
      });

      const response = await result.response;
      
      const images = response.candidates?.map(c => {
        const part = c.content.parts.find(p => p.inlineData);
        return part ? `data:${part.inlineData.mimeType};base64,${part.inlineData.data}` : null;
      }).filter(Boolean) as string[];

      return { imageURLs: images };

    } catch (err) {
      return { error: err instanceof Error ? err.message : String(err) };
    }
  }
}