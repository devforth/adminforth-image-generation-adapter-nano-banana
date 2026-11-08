export interface AdapterOptions {
  /**
   * Gemini API key. Go to https://aistudio.google.com/apikey and create a key.
   * Paste value in your .env file GEMINI_API_KEY=your_key
   * Set nanoBananaApiKey: process.env.GEMINI_API_KEY to access it
   */
  nanoBananaApiKey: string;

  /**
   * Model name. Go to https://aistudio.google.com/docs/models, select model and copy name.
   * Default is `gemini-nano-banana-2.1`. Use e.g. `gemini-3-pro-image` for more powerful model.
   */
  model?: string;

  /**
   * Extra parameters merged into the Gemini `generationConfig` of every generation request.
   * Check here for params https://aistudio.google.com/docs/image-generation
   */
  extraParams?: {
    [key: string]: any;
  };

  /**
   * Hosts which input images are allowed to be downloaded from, e.g. `['my-bucket.s3.amazonaws.com']`.
   * When not set, images are downloaded from any host.
   * Entries are matched case-insensitively, `example.com` matches that host only, `.example.com` (or `*.example.com`)
   * matches it and its subdomains.
   */
  attachImagesAllowedHosts?: string[];

}
