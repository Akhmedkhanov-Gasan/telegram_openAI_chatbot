import os
import asyncio
from collections import defaultdict, deque

from openai import AsyncOpenAI
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram import F
from aiogram.enums import ChatAction

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

chat_history = defaultdict(lambda: deque(maxlen=5))

async def show_typing(bot: Bot, chat_id: int):
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

async def send_in_chunks(message: Message, text: str, chunk_size: int = 3500):
    for i in range(0, len(text), chunk_size):
        await message.answer(text[i:i+chunk_size])

@dp.message(F.text)
async def ask_local_llm(message: Message):
    user_id = message.chat.id
    prompt = message.text.strip()
    if not prompt:
        return

    chat_history[user_id].append({"role": "user", "content": prompt})

    task = asyncio.create_task(show_typing(message.bot, message.chat.id))
    try:
        resp = await ollama_client.chat.completions.create(
            model="qwen2.5:14b-instruct-q5_K_M",
            messages=list(chat_history[user_id]),
            temperature=0.7,
            max_tokens=800,
        )
        text = resp.choices[0].message.content

        chat_history[user_id].append({"role": "assistant", "content": text})

        await send_in_chunks(message, text or "(empty)")
    except Exception as e:
        await message.answer(f"Local LLM error: {e}")
    finally:
        task.cancel()

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
