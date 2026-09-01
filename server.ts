import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Middleware for parsing large JSON payloads (base64 image frames)
app.use(express.json({ limit: "30mb" }));

// Lazy Google GenAI Client
let aiClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn("GEMINI_API_KEY environment variable is not set.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey || "",
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// Health check endpoint
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    hasApiKey: Boolean(process.env.GEMINI_API_KEY),
    timestamp: Date.now(),
  });
});

/**
 * High-accuracy Banknote and Scene Analysis Endpoint
 */
app.post("/api/analyze-vision", async (req, res) => {
  try {
    const {
      imageBase64,
      mode = "scene",
      query = "",
      targetItem = "",
      knownPeople = [],
      language = "en",
    } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: "Missing imageBase64 in request body" });
    }

    // Extract pure base64 data and mime type
    let mimeType = "image/jpeg";
    let pureBase64 = imageBase64;
    if (imageBase64.startsWith("data:")) {
      const match = imageBase64.match(/^data:([^;]+);base64,(.+)$/);
      if (match) {
        mimeType = match[1];
        pureBase64 = match[2];
      } else {
        const commaIndex = imageBase64.indexOf(",");
        if (commaIndex !== -1) {
          pureBase64 = imageBase64.substring(commaIndex + 1);
        }
      }
    }

    if (mimeType.includes("png")) {
      mimeType = "image/png";
    } else if (mimeType.includes("webp")) {
      mimeType = "image/webp";
    } else {
      mimeType = "image/jpeg";
    }

    const hasApiKey = Boolean(process.env.GEMINI_API_KEY);

    if (!hasApiKey) {
      const isHindi = language === "hi";
      const noKeySummary = isHindi
        ? "कृपया सेटिंग्स में जेमिनी एपीआई की (GEMINI_API_KEY) सेट करें ताकि एआई दृष्टि नोट व वस्तुओं को सटीक रूप से पहचान सके।"
        : "Gemini API key is not configured. Please add GEMINI_API_KEY in settings to enable live AI banknote and vision analysis.";
      return res.json({
        mode,
        spokenSummary: noKeySummary,
        fullDetails: noKeySummary,
        hazards: [],
        detectedPeople: { count: 0, descriptions: [] },
        keyObjects: [],
        timestamp: Date.now(),
        imageThumbnail: imageBase64.length < 500000 ? imageBase64 : undefined,
      });
    }

    // Compose System Instructions tailored to accessibility and high accuracy currency identification
    let langInstruction = "Language: Provide output strictly in natural, clear English.";
    if (language === "hi") {
      langInstruction = "Language: You MUST generate 'spokenSummary' and 'fullDetails' in fluent, polite, grammatically correct HINDI (Devanagari script हिन्दी) suitable for Text-To-Speech audio playback.";
    } else if (language === "es") {
      langInstruction = "Language: You MUST generate 'spokenSummary' and 'fullDetails' in fluent SPANISH (Español).";
    } else if (language === "fr") {
      langInstruction = "Language: You MUST generate 'spokenSummary' and 'fullDetails' in fluent FRENCH (Français).";
    } else if (language === "de") {
      langInstruction = "Language: You MUST generate 'spokenSummary' and 'fullDetails' in fluent GERMAN (Deutsch).";
    } else if (language === "ja") {
      langInstruction = "Language: You MUST generate 'spokenSummary' and 'fullDetails' in fluent JAPANESE (日本語).";
    }

    const systemInstruction = `You are Lumina Vision, an advanced, compassionate assistive AI vision assistant designed for visually impaired and blind users.
Your goal is to provide 100% truthful, precise, spatial, and descriptive audio narration of whatever camera view or image is presented to you.

CRITICAL ACCURACY RULES FOR BANKNOTES & CURRENCY:
1. Examine the image carefully before stating any banknote denomination. Look at the exact printed numerals (e.g., "100", "500", "200", "50", "20", "10", "2000" in Latin or Devanagari numerals), the central portrait, and the distinctive color theme:
   - Indian Rupee (INR) Denomination Guide:
     • ₹10 Note: Chocolate Brown color, Sun Temple Konark motif.
     • ₹20 Note: Greenish-Yellow color, Ellora Caves motif.
     • ₹50 Note: Fluorescent Blue color, Hampi Stone Chariot motif.
     • ₹100 Note: Lavender / Pale Purple color, Rani ki Vav stepwell motif. Numeral "100" and "१००". CRITICAL: DO NOT confuse ₹100 with ₹500 or ₹200!
     • ₹200 Note: Bright Orange / Yellow color, Sanchi Stupa motif. Numeral "200" and "२००".
     • ₹500 Note: Stone Grey / Olive Green tone, Red Fort motif. Numeral "500" and "५००".
     • ₹2000 Note: Magenta / Deep Pink tone, Mangalyaan motif. Numeral "2000".
   - US Dollar (USD) Denomination Guide:
     • $1 (Washington), $5 (Lincoln), $10 (Hamilton), $20 (Jackson), $50 (Grant), $100 (Franklin).
   - Euro (EUR), British Pound (GBP), and other currencies: inspect the numeral and currency symbols clearly.
2. If multiple banknotes or coins are visible (e.g., a stack, fanned out, or laid on a table):
   - Count the quantity of EACH denomination individually.
   - Calculate the arithmetic TOTAL monetary sum.
   - State the total amount first, then list the breakdown clearly (e.g. "Total 1,300 Rupees across 4 notes: Two 500 Rupee notes, one 200 Rupee note, and one 100 Rupee note.").
3. NEVER guess or hallucinate a 500 Rupee note or 5 Dollar note if a 100 Rupee note or other note is present. If the image is blurry or partially obscured, state what is visible with confidence and advise how to hold the note.

SAFETY & SPATIAL RULES:
- Use clock positions (12 o'clock = straight ahead, 9 o'clock = left, 3 o'clock = right) and approximate distance estimates (feet/meters) for all key objects and people.
- Prioritize urgent hazards (steps, drop-offs, low hanging objects, hot surfaces, obstacles in walking path).
${langInstruction}`;

    let userPrompt = "";

    switch (mode) {
      case "text":
        userPrompt = `Perform precise OCR reading of all visible text in this image for a visually impaired user.
1. Transcribe all text accurately in reading order.
2. If this is a medicine label, grocery packaging, menu, or official document, highlight key details (active ingredients, dosage, expiry dates, prices, allergies, warnings).
3. Formulate a 'spokenSummary' that clearly announces the primary title/heading and vital text.`;
        break;

      case "faces":
        const knownPeopleInfo = Array.isArray(knownPeople) && knownPeople.length > 0
          ? `User's registered contacts: ${JSON.stringify(knownPeople.map((p: { name: string; relationship: string }) => `${p.name} (${p.relationship})`))}. Match if applicable.`
          : "";
        userPrompt = `Analyze people and faces in this camera view for a visually impaired user.
${knownPeopleInfo}
1. Count the number of people and describe spatial positions (e.g. 'One person directly in front at 12 o'clock, 4 feet away').
2. Describe expressions, emotional state (smiling, talking, attentive), estimated age, and distinctive clothing.
3. Provide a natural 'spokenSummary'.`;
        break;

      case "find":
        userPrompt = `Locate the target item: "${targetItem || query || "object"}" in the image for a visually impaired user.
1. Determine if the item is present.
2. If found, set foundTarget.found=true, specify exact clock direction and distance estimate, and describe nearby tactile landmarks to reach it safely.
3. If not found, set foundTarget.found=false and guide where to aim next.`;
        break;

      case "currency_products":
        userPrompt = `Analyze all cash banknotes, coins, grocery products, or store items in this image.
1. If ANY banknotes or coins are visible:
   - Identify the exact currency (e.g. Indian Rupee ₹, US Dollar $, Euro €) and denomination (e.g. ₹100, ₹500, ₹200, ₹50, $20).
   - Count every single banknote and coin in the frame.
   - Calculate the EXACT total sum.
   - Fill 'currencyOrProductDetails' with:
     * type: 'currency'
     * denominationOrTitle: e.g. "₹100 (Indian Rupee Note)" or "₹1,300 Total (4 Banknotes)"
     * totalSum: e.g. "₹100" or "₹1,300"
     * totalNotesCount: total number of notes
     * detectedNotes: array of [{ denomination: "₹100", currency: "Indian Rupee (INR)", count: 1, subtotal: "₹100", position: "Center of view" }]
     * vitalInfo: visible security features (motifs, watermarks, condition)
   - In 'spokenSummary', state the denomination and total amount clearly and directly.
2. If this is a grocery or retail product:
   - Identify brand name, product variant, weight/volume, and expiration date.`;
        break;

      case "continuous":
        userPrompt = `Continuous walking navigation check for a visually impaired user.
1. Flag any immediate obstacle, trip hazard, stairs, drop-offs, or moving traffic in the pathway.
2. Provide a 1-2 sentence concise 'spokenSummary' stating if path is clear or if immediate caution is required.`;
        break;

      case "qa":
        userPrompt = `A visually impaired user is asking: "${query || "What is in front of me?"}"
Answer directly, concisely, and accurately based strictly on what is in this image. If asking about money, banknotes, or denominations, verify the exact numerals and count before answering.`;
        break;

      case "scene":
      default:
        userPrompt = `Describe the overall scene and environment for a visually impaired user.
1. Environment type (room, outdoor, sidewalk, store, kitchen).
2. Pathway and obstacles.
3. Prominent objects and people with clock positions and distance.
4. If cash, banknotes, or currency are visible, identify the exact notes and calculate the total sum.
5. Provide a 2-3 sentence engaging 'spokenSummary' followed by full details.`;
        break;
    }

    const imagePart = {
      inlineData: {
        mimeType,
        data: pureBase64,
      },
    };

    const baseSchema = {
      type: Type.OBJECT,
      properties: {
        spokenSummary: {
          type: Type.STRING,
          description: "Clear, natural 1-3 sentence narration formulated to be spoken aloud by TTS.",
        },
        fullDetails: {
          type: Type.STRING,
          description: "Comprehensive descriptive breakdown of the scene, layout, text, or elements.",
        },
        detectedText: {
          type: Type.STRING,
          description: "Extracted text or OCR in reading order if visible.",
        },
        hazards: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              type: {
                type: Type.STRING,
                description: "obstacle, drop_off, low_ceiling, moving_object, spill, traffic, or other",
              },
              description: { type: Type.STRING },
              urgency: {
                type: Type.STRING,
                description: "immediate, caution, or advisory",
              },
              direction: { type: Type.STRING },
            },
            required: ["type", "description", "urgency"],
          },
        },
        detectedPeople: {
          type: Type.OBJECT,
          properties: {
            count: { type: Type.INTEGER },
            descriptions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            recognizedNames: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
            expressions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
            },
          },
        },
        foundTarget: {
          type: Type.OBJECT,
          properties: {
            found: { type: Type.BOOLEAN },
            itemName: { type: Type.STRING },
            locationDescription: { type: Type.STRING },
            clockDirection: { type: Type.STRING },
            distance: { type: Type.STRING },
          },
        },
        keyObjects: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              name: { type: Type.STRING },
              clockPosition: { type: Type.STRING },
              distanceEstimate: { type: Type.STRING },
              confidence: { type: Type.STRING },
            },
            required: ["name", "clockPosition", "distanceEstimate"],
          },
        },
        currencyOrProductDetails: {
          type: Type.OBJECT,
          properties: {
            type: { type: Type.STRING },
            denominationOrTitle: { type: Type.STRING },
            totalSum: { type: Type.STRING },
            totalNotesCount: { type: Type.INTEGER },
            detectedNotes: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  denomination: { type: Type.STRING },
                  currency: { type: Type.STRING },
                  count: { type: Type.INTEGER },
                  subtotal: { type: Type.STRING },
                  position: { type: Type.STRING },
                  condition: { type: Type.STRING },
                },
                required: ["denomination", "count"],
              },
            },
            vitalInfo: { type: Type.STRING },
          },
        },
      },
      required: ["spokenSummary", "fullDetails"],
    };

    const ai = getGenAI();
    let response: any = null;
    let lastError: any = null;

    // Use official active models with resilient fallbacks
    const modelsToTry = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.7-flash"];

    for (const modelName of modelsToTry) {
      // Try up to 2 attempts with a short backoff for transient 503/429
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          if (attempt > 0) {
            await new Promise((resolve) => setTimeout(resolve, 800 * attempt));
          }
          response = await ai.models.generateContent({
            model: modelName,
            contents: [imagePart, userPrompt],
            config: {
              systemInstruction,
              temperature: 0.1,
              maxOutputTokens: 600,
              responseMimeType: "application/json",
              responseSchema: baseSchema,
            },
          });
          if (response && response.text) {
            break;
          }
        } catch (err: any) {
          lastError = err;
          const status = err?.status || err?.code || "";
          const msg = err?.message || "";
          console.warn(`Vision model ${modelName} (attempt ${attempt + 1}) failed [${status}]:`, msg);
          
          // If it's a 404 (model not found) or invalid arg, don't retry same model
          if (status === 404 || status === 400) {
            break;
          }
        }
      }

      if (response && response.text) {
        break;
      }
    }

    if (!response || !response.text) {
      throw lastError || new Error("Unable to analyze image. Please try again.");
    }

    const textOutput = response.text || "{}";
    let parsedResult: any = {};
    try {
      parsedResult = JSON.parse(textOutput);
    } catch {
      parsedResult = {
        spokenSummary: textOutput,
        fullDetails: textOutput,
      };
    }

    const payload = {
      mode,
      spokenSummary: parsedResult.spokenSummary || (language === "hi" ? "चित्र का विवरण उपलब्ध नहीं है।" : "No description available."),
      fullDetails: parsedResult.fullDetails || parsedResult.spokenSummary || "",
      detectedText: parsedResult.detectedText || "",
      hazards: parsedResult.hazards || [],
      detectedPeople: parsedResult.detectedPeople || { count: 0, descriptions: [] },
      foundTarget: parsedResult.foundTarget || undefined,
      keyObjects: parsedResult.keyObjects || [],
      currencyOrProductDetails: parsedResult.currencyOrProductDetails || undefined,
      timestamp: Date.now(),
      imageThumbnail: imageBase64.length < 500000 ? imageBase64 : undefined,
    };

    return res.json(payload);
  } catch (error: unknown) {
    console.error("Error analyzing vision:", error);
    const errorMessage = error instanceof Error ? error.message : "Failed to analyze image";
    const isRateLimit = errorMessage.includes("429") || errorMessage.includes("quota") || errorMessage.includes("RESOURCE_EXHAUSTED");
    const isHindi = req.body?.language === "hi";
    
    let spokenSummary = isHindi
      ? "क्षमा करें, चित्र का विश्लेषण करने में समस्या हुई। कृपया नोट या दृश्य को कैमरे के सामने स्थिर रखें और पुनः प्रयास करें।"
      : "Sorry, I had trouble analyzing the image. Please hold the camera steady and scan again.";
      
    if (isRateLimit) {
      spokenSummary = isHindi
        ? "एआई सेवा अभी व्यस्त है। कृपया कुछ सेकंड प्रतीक्षा करके पुनः स्कैन करें।"
        : "The AI service is momentarily busy. Please wait a few seconds and scan again.";
    }

    return res.status(500).json({
      error: errorMessage,
      spokenSummary,
      fullDetails: errorMessage,
      timestamp: Date.now(),
    });
  }
});

/**
 * Natural Language Q&A on Live Camera Frame
 */
app.post("/api/ask-question", async (req, res) => {
  try {
    const { imageBase64, question, language = "en" } = req.body;
    if (!imageBase64 || !question) {
      return res.status(400).json({ error: "Missing imageBase64 or question" });
    }

    let mimeType = "image/jpeg";
    let pureBase64 = imageBase64;
    if (imageBase64.startsWith("data:")) {
      const match = imageBase64.match(/^data:([^;]+);base64,(.+)$/);
      if (match) {
        mimeType = match[1];
        pureBase64 = match[2];
      } else {
        const commaIndex = imageBase64.indexOf(",");
        if (commaIndex !== -1) {
          pureBase64 = imageBase64.substring(commaIndex + 1);
        }
      }
    }

    if (mimeType.includes("png")) {
      mimeType = "image/png";
    } else if (mimeType.includes("webp")) {
      mimeType = "image/webp";
    } else {
      mimeType = "image/jpeg";
    }

    const hasApiKey = Boolean(process.env.GEMINI_API_KEY);
    if (!hasApiKey) {
      const isHindi = language === "hi";
      const noKeyMsg = isHindi
        ? "कृपया सेटिंग्स में GEMINI_API_KEY सेट करें।"
        : "Please configure GEMINI_API_KEY in settings to ask questions about live camera images.";
      return res.json({ answer: noKeyMsg, timestamp: Date.now() });
    }

    let langPrompt = "Answer in clear, concise English for a blind or visually impaired user.";
    if (language === "hi") {
      langPrompt = "You MUST respond in natural, polite, grammatically correct HINDI (हिन्दी) in Devanagari script suitable for Text-To-Speech.";
    }

    const promptText = `The user is asking: "${question}".
Analyze what is visible in the image to answer accurately and helpfully.
If the question is about currency, cash, or banknotes:
- Inspect the exact numeral (e.g. 100, 200, 500, 50, 20, 10 for INR or USD/EUR) and color.
- If it is a ₹100 note (lavender / Rani ki Vav), announce it as 100 Rupees (₹100), NOT ₹500!
- Count each note and provide the exact total sum.
${langPrompt}`;

    const ai = getGenAI();
    let response: any = null;
    const modelsToTry = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.7-flash"];

    for (const modelName of modelsToTry) {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          if (attempt > 0) {
            await new Promise((resolve) => setTimeout(resolve, 800 * attempt));
          }
          response = await ai.models.generateContent({
            model: modelName,
            contents: [
              {
                inlineData: {
                  mimeType,
                  data: pureBase64,
                },
              },
              promptText,
            ],
            config: {
              systemInstruction: `You are Lumina Vision, an assistive AI for visually impaired users. Provide direct, truthful, accurate answers. ${langPrompt}`,
              temperature: 0.1,
              maxOutputTokens: 350,
            },
          });
          if (response && response.text) break;
        } catch (err: any) {
          console.warn(`Ask question model ${modelName} failed:`, err?.message || err);
        }
      }
      if (response && response.text) break;
    }

    const answer = response?.text || (
      language === "hi"
        ? "माफ़ कीजिए, मैं इस प्रश्न का उत्तर नहीं दे सका। कृपया पुनः प्रयास करें।"
        : "Sorry, I could not answer your question. Please try asking again."
    );

    return res.json({ answer, timestamp: Date.now() });
  } catch (error: unknown) {
    console.error("Error asking question:", error);
    const msg = error instanceof Error ? error.message : "An error occurred";
    return res.status(500).json({ error: msg, answer: "Sorry, I could not process your question." });
  }
});

// Vite Middleware for Dev / Static serving for Prod
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Lumina Vision Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
