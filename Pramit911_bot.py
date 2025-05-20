from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import re
from io import BytesIO
import torch
from diffusers import DiffusionPipeline
import asyncio

# Set your Gemini API key here
GEMINI_API_KEY = "AIzaSyAP2XFyVWMpqiwHIiFdpAYVhNGeqU_x42w"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY

# Load the pipeline globally (loads once at startup)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant="fp16" if device == "cuda" else None
)
pipe.to(device)

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm your bot.")

def get_gemini_response(user_message: str) -> str:
    """Send a message to Google Gemini API and return the response text."""
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": user_message}]}]
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[Gemini API error: {e}]"

def strip_markdown(md_text: str) -> str:
    """Remove basic Markdown formatting for plain text output."""
    # Remove headers, bold, italics, inline code, blockquotes, lists, etc.
    text = re.sub(r'[#*_`>\-\[\]()>~]', '', md_text)  # Remove markdown symbols
    text = re.sub(r'\n{2,}', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'\s{2,}', ' ', text)   # Collapse multiple spaces
    return text.strip()

# Echo message handler (now chats with Gemini)
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    sent_message = await update.message.reply_text("Thinking...")
    gemini_reply = get_gemini_response(user_message)
    plain_reply = strip_markdown(gemini_reply)
    try:
        await context.bot.edit_message_text(
            chat_id=sent_message.chat_id,
            message_id=sent_message.message_id,
            text=plain_reply
        )
    except Exception as e:
        await update.message.reply_text(plain_reply)

async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please provide a prompt. Usage: /image <prompt>")
        return
    prompt = ' '.join(context.args)
    await update.message.reply_text("Generating image (this may take a while)...")

    def generate_image(prompt):
        result = pipe(prompt=prompt)
        image = result.images[0]
        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        return image_bytes

    loop = asyncio.get_event_loop()
    image_bytes = await loop.run_in_executor(None, generate_image, prompt)
    image_bytes.name = 'generated.png'
    await update.message.reply_photo(photo=image_bytes)

# Main function
def main():
    app = ApplicationBuilder().token("7090275475:AAGlNesSebvtIHDtUy-NpLPoI6kEqIXfVac").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("image", image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    print("Bot is running...")
    app.run_polling()

# To run the bot
if __name__ == '__main__':
    main()
