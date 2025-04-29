import asyncio
import gradio as gr
import os
import time
import shutil

from app.agent.manus import Manus
from app.logger import logger
from loguru import logger as loguru_logger

WORKSPACE_DIR = "workspace"


class GradioLoggerSink:

    """A custom sink to capture the logs from openmanus and stream them live into Gradio."""

    def __init__(self):
        self.queue = asyncio.Queue()

    def write(self, message):
        asyncio.create_task(self.queue.put(message.strip()))

    async def generator(self):
        while True:
            log = await self.queue.get()
            yield log


async def run_and_stream(prompt_text, chat_history):
    global agent
    if agent is None:
        agent = await Manus.create()

    existing_files = set(os.listdir(WORKSPACE_DIR)) if os.path.exists(WORKSPACE_DIR) else set()

    sink = GradioLoggerSink()
    sink_id = loguru_logger.add(sink, format="{message}")

    try:
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": prompt_text})
        yield chat_history, ""

        task = asyncio.create_task(agent.run(prompt_text))

        async for log in sink.generator():
            if log.strip():
                chat_history.append({"role": "assistant", "content": log})
                yield chat_history, ""
            if task.done():
                break

        await task

        while not sink.queue.empty():
            log = await sink.queue.get()
            if log.strip():
                chat_history.append({"role": "assistant", "content": log})
                yield chat_history, ""

        # After Manus finishes the task this it to detect new files in the workspace directory and render it in the gradio app
        if os.path.exists(WORKSPACE_DIR):
            new_files = set(os.listdir(WORKSPACE_DIR)) - existing_files
            for file_name in new_files:
                file_path = os.path.join(WORKSPACE_DIR, file_name)
                if file_name.endswith(".md"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    chat_history.append({
                        "role": "assistant",
                        "content": f"### 📄 Markdown File Created:\n\n{content}"
                    })
                    yield chat_history, ""
                elif file_name.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    chat_history.append({
                        "role": "assistant",
                        "content": f"📝 Text File Created:\n\n```\n{content}\n```"
                    })
                    yield chat_history, ""

    finally:
        loguru_logger.remove(sink_id)
        await agent.cleanup()
        agent = None

def clear_chat():
    return [], ""

# A funtion to download the workspace folder
def package_workspace():
    if os.path.exists(WORKSPACE_DIR):
        shutil.make_archive('workspace_backup', 'zip', WORKSPACE_DIR)
        return "workspace_backup.zip"
    else:
        return None


# Global Manus agent
agent = None

# Gradio UI
with gr.Blocks(theme=gr.themes.Default()) as app:
    gr.Markdown("# 🤖 OpenManus AI Chatbot", elem_id="title")

    chatbot = gr.Chatbot(label="Chat with OpenManus", height=600, type="messages")

    with gr.Row():
        prompt = gr.Textbox(
            placeholder="Type your question here...",
            lines=2,
            label="Your Prompt"
        )
        submit_btn = gr.Button("🚀 Submit")

    with gr.Row():
        clear_btn = gr.Button("🧹 Clear Chat")
        download_workspace = gr.Button("📥 Download Workspace")

    state = gr.State([])

    submit_btn.click(
        run_and_stream,
        inputs=[prompt, state],
        outputs=[chatbot, prompt]
    )
    clear_btn.click(
        clear_chat,
        inputs=[],
        outputs=[chatbot, prompt]
    )
    download_workspace.click(
        package_workspace,
        inputs=[],
        outputs=gr.File(label="Workspace.zip")
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
