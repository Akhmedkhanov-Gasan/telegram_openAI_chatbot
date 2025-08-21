import os
import asyncio

from huggingface_hub import AsyncInferenceClient

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import F


load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
HF_TOKEN = os.getenv("HF_TOKEN")


bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
hf = AsyncInferenceClient(
    model="google/gemma-2-2b-it",
    token=HF_TOKEN,
    timeout=60.0,
)

@dp.message(Command('start'))
async def start_handler(message: Message):
    await message.answer(f'Hello there! {message.from_user.first_name}')

@dp.message(Command('help'))
async def help_handler(message: Message):
    await message.answer('Type "/start" to start the bot. ')

@dp.message(F.text, ~F.text.startswith('/'))
async def echo_handler(message: Message):
    await message.answer(message.text)

@dp.message(Command("ask"))
async def ask_hf_handler(message: Message):
    prompt = message.text.removeprefix("/ask").strip()
    if not prompt:
        await message.answer("Usage: /ask your question")
        return
    if not HF_TOKEN:
        await message.answer("HF token is missing on the server.")
        return
    try:
        result = await hf.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        text = result.choices[0].message["content"]
        await message.answer(text or "(empty)")
    except Exception as e:
        await message.answer(f"HF error: {e}")


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await hf.aclose()

if __name__ == '__main__':
    asyncio.run(main())
