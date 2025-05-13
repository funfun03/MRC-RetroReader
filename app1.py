import gradio as gr
import os
from retro_reader import RetroReader

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Load RetroReader từ thư viện
def from_library():
    from retro_reader import constants as C
    return C, RetroReader

C, RetroReader = from_library()

# Load mô hình Roberta từ file cấu hình
def load_model(config_path):
    return RetroReader.load(config_file=config_path)

# Đường dẫn đến file config
model_roberta = load_model("E:/WORKSPACE/RetroReader/MRC-RetroReader/configs/inference_en_roberta.yaml")

def retro_reader_demo(query, context):
    outputs = model_roberta(query=query, context=context, return_submodule_outputs=True)
    answer = outputs[0]["id-01"] if outputs[0]["id-01"] else "No answer found"
    return answer

# Giao diện Gradio
iface = gr.Interface(
    fn=retro_reader_demo,
    inputs=[
        gr.Textbox(label="Query", placeholder="Type your query here..."),
        gr.Textbox(label="Context", placeholder="Provide the context here...", lines=10)
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Roberta RetroReader Demo",
    description="This interface uses the Roberta-based RetroReader model for reading comprehension."
)

if __name__ == "__main__":
    iface.launch(share=True)

