import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def get_civic_info(query):
    """Generates information on a given civic topic."""
    prompt = f"Provide a comprehensive overview of the following civic topic. Include key details, responsibilities, and relevant resources.\n\nCivic Topic: {query}\n\nInformation:\n\n**IMPORTANT: This is for informational purposes only. Please verify with official government sources.**\n\nAnalysis:"
    return generate_response(prompt, max_length=1200)

def analyze_sentiment_and_update_dashboard(feedback_text, current_positive, current_neutral, current_negative, current_total):
    """Analyzes sentiment and updates the dashboard counts in a single function call."""
    prompt = f"Analyze the sentiment of the following citizen feedback and classify it as 'Positive', 'Neutral', or 'Negative'.\n\nFeedback: {feedback_text}\n\nSentiment:"
    response = generate_response(prompt, max_length=100)
    sentiment = response.lower().strip()
    
    new_positive, new_neutral, new_negative = current_positive, current_neutral, current_negative
    if "positive" in sentiment:
        new_positive += 1
        result = "Positive"
    elif "negative" in sentiment:
        new_negative += 1
        result = "Negative"
    else:
        new_neutral += 1
        result = "Neutral"
        
    new_total = current_total + 1
    
    return result, new_positive, new_neutral, new_negative, new_total

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Citizen AI Assistant")
    gr.Markdown("**Disclaimer: This AI is for informational and creative purposes. It does not provide professional legal, financial, or medical advice.**")

    with gr.Tabs():
        with gr.TabItem("Civic Information"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Enter a civic topic",
                        placeholder="e.g., how to register to vote, local recycling laws, requirements for a business license...",
                        lines=4
                    )
                    info_btn = gr.Button("Get Information")

                with gr.Column():
                    info_output = gr.Textbox(label="Information & Analysis", lines=20)

            info_btn.click(get_civic_info, inputs=query_input, outputs=info_output)

        with gr.TabItem("Citizen Sentiment Analysis"):
            gr.Markdown("Analyze citizen feedback to understand public opinion.")
            with gr.Row():
                with gr.Column():
                    feedback_input = gr.Textbox(
                        label="Enter Citizen Feedback",
                        placeholder="e.g., The new bus route is fantastic! It's so efficient.",
                        lines=5
                    )
                    analyze_btn = gr.Button("Analyze Sentiment")

                with gr.Column():
                    sentiment_output = gr.Textbox(
                        label="Sentiment Result",
                        interactive=False
                    )

        with gr.TabItem("Dynamic Dashboard"):
            gr.Markdown("Visualize real-time sentiment data and key metrics.")
            with gr.Row():
                with gr.Column():
                    total_feedback = gr.Number(label="Total Feedback Submissions", value=0, interactive=False)
                    positive_count = gr.Number(label="Positive Sentiment", value=0, interactive=False)
                    neutral_count = gr.Number(label="Neutral Sentiment", value=0, interactive=False)
                    negative_count = gr.Number(label="Negative Sentiment", value=0, interactive=False)
        
    analyze_btn.click(
        analyze_sentiment_and_update_dashboard,
        inputs=[feedback_input, positive_count, neutral_count, negative_count, total_feedback],
        outputs=[sentiment_output, positive_count, neutral_count, negative_count, total_feedback]
    )

app.launch(share=True)
