import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    // 1. Get the file from the incoming frontend request
    const file = formData.get('file');

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // 2. Prepare the data for the Python ML API (Switching key to 'image')
    const mlFormData = new FormData();
    mlFormData.append('image', file);

    // 3. Connect to the Flask server running on port 8000
    const ML_API_URL = 'http://127.0.0.1:8000/predict'; 

    const response = await fetch(ML_API_URL, {
      method: 'POST',
      body: mlFormData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ML API Error (${response.status}): ${errorText}`);
    }

    const result = await response.json();

    // 4. Return the prediction result back to the browser
    return NextResponse.json({
      label: result.label || result.class_name || "Unknown",
      confidence: result.confidence ?? 0,
      all_predictions: result.all_predictions || [],
      processing_time: result.processing_time || 0,
      success: true
    });

  } catch (error: any) {
    console.error("ML Bridge Error:", error);
    return NextResponse.json({ 
      error: "ML Service Unavailable", 
      details: error.message 
    }, { status: 500 });
  }
}