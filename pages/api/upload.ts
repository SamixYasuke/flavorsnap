import { NextApiRequest, NextApiResponse } from 'next';
import * as ort from 'onnxruntime-node';
import sharp from 'sharp';
import path from 'path';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    // 1. Get image from request (assuming buffer/base64)
    const { image } = req.body; 
    const buffer = Buffer.from(image, 'base64');

    // 2. Preprocess with Sharp
    const processedImage = await sharp(buffer)
      .resize(224, 224)
      .toFormat('raw')
      .toBuffer();

    // 3. Convert to Float32 Tensor & Normalize
    const floatData = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < processedImage.length; i++) {
      // Basic normalization logic here
      floatData[i] = processedImage[i] / 255.0; 
    }
    
    const tensor = new ort.Tensor('float32', floatData, [1, 3, 224, 224]);

    // 4. Run Inference
    const modelPath = path.join(process.cwd(), 'models/resnet18.onnx');
    const session = await ort.InferenceSession.create(modelPath);
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);

    // 5. Get Prediction
    const output = results[session.outputNames[0]].data as Float32Array;
    const predictedClass = output.indexOf(Math.max(...Array.from(output)));

    return res.status(200).json({ label: `Class ${predictedClass}`, confidence: Math.max(...Array.from(output)) });

  } catch (error) {
    console.error(error);
    return res.status(500).json({ error: 'Inference failed' });
  }
}