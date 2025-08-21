import os
import asyncio

from openai import AsyncOpenAI

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import F


load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
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
async def ask_local_llm(message: Message):
    prompt = message.text.removeprefix("/ask").strip()
    if not prompt:
        await message.answer("Usage: /ask your question")
        return
    try:
        resp = await ollama_client.chat.completions.create(
            model="llama3.1:8b-instruct-q4_K_M",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        text = resp.choices[0].message.content
        await message.answer(text[:4096] or "(empty)")
    except Exception as e:
        await message.answer(f"Local LLM error: {e}")


async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
