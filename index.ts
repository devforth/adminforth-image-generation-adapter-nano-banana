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

  outputImagesMaxCountSupported(): number {
    return 4;
  }

  outputDimensionsSupported(): string[] {
    return ['1024x1024', '1440x1024', '1024x1440'];
  }

  inputFileExtensionSupported(): string[] {
    return ['png', 'jpg', 'jpeg'];
  }

  private assertHostAllowed(url: string): void {
    const allowedHosts = this.options.attachImagesAllowedHosts;
    if (!allowedHosts?.length) {
      return;
    }
    let hostname: string;
    try {
      hostname = new URL(url).hostname.toLowerCase().replace(/\.$/, '');
    } catch {
      throw new Error(`Invalid attachment URL: ${url}`);
    }
    const allowed = allowedHosts.some((raw) => {
      const entry = (raw || '').trim().toLowerCase().replace(/^\*/, '').replace(/\.$/, '');
      if (!entry) {
        return false;
      }
      return entry.startsWith('.') ? (hostname === entry.slice(1) || hostname.endsWith(entry)) : hostname === entry;
    });
    if (!allowed) {
      throw new Error(`Attachment host "${hostname}" is not in attachImagesAllowedHosts`);
    }
  }

  private async urlToGenerativePart(url: string) {
    this.assertHostAllowed(url);
    const response = await fetch(url);
    if (response.status >= 300 && response.status < 400) {
      throw new Error(`Failed to fetch image from URL: ${url}, status: ${response.status}. Redirects are not allwed.`);
    }
    const buffer = await response.arrayBuffer();
    const contentType = response.headers.get("content-type") || "image/jpeg";
    return {
      inlineData: {
        data: Buffer.from(buffer).toString("base64"),
        mimeType: contentType,
      },
    };
  }

  private mapSizeToImageConfig(size: string) {
    const [width, height] = size.split('x').map(Number);

    const knownRatios: Record<string, string> = {
      '1:1': '1:1', '16:9': '16:9', '9:16': '9:16',
      '4:3': '4:3', '3:4': '3:4', '3:2': '3:2', '2:3': '2:3',
      '21:9': '21:9', '4:1': '4:1', '8:1': '8:1'
    };
    const ratioKey = Object.keys(knownRatios).find(k => {
      const [rw, rh] = k.split(':').map(Number);
      return Math.abs(rw / rh - width / height) < 0.05; 
    });
    const aspectRatio = ratioKey ? knownRatios[ratioKey] : '1:1';
 
    const maxDim = Math.max(width, height);
    let imageSize = '1K';
    if (maxDim <= 1024) imageSize = '1K';
    else if (maxDim <= 2048) imageSize = '2K';
    else if (maxDim <= 4096) imageSize = '4K';
    else imageSize = '4K';

    return { aspectRatio, imageSize };
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

      const { aspectRatio, imageSize } = this.mapSizeToImageConfig(size);

      const generationConfig: any = {
        candidateCount: n,
        imageConfig: { aspectRatio, imageSize },
        responseModalities: ['Image'],
      };

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