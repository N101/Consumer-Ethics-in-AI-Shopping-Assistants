from fpdf import FPDF
import os
import subprocess
import pandas as pd
import json

from config.configuration import DATA_FOLDER_PATH, SYSTEM_PROMPT, STATE_FILE

# Load the state from the file
def load_state(prefix) -> tuple:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            if prefix in state.keys():
                return prefix, state[prefix]
    return "", 1

# Save the state to the file
def save_state(prefix, counter):
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        state[prefix] = counter
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)

def init_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=25.4, top=25.4, right=25.4)  # Set A4 default margins (25.4 mm)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    pdf.set_y(25.4)  # Ensure the top margin is set correctly
    return pdf


def create_pdf_report(model: str, llm: str, prefix: str, data_list: pd.DataFrame, images: list):
    # Initialize the global variables
    global NAME, COUNTER
    NAME, COUNTER = load_state(prefix)

    pdf = init_pdf()

    if NAME == prefix:
        COUNTER += 1
        prefix = f"{prefix}_{COUNTER}"
    else:
        NAME = prefix
        prefix = f"{prefix}_1"
    
    # Title
    pdf.set_font("Times", 'B', 16)
    pdf.cell(160, 10, f"Evaluation Report for {model.upper()} ({llm})", ln=True, align='C')
    pdf.ln(10)
    
    # Parameters
    pdf.set_font("Times", size=12)
    pdf.cell(160, 5, f"Model: {model}      (prefix: {prefix})", ln=True)
    pdf.cell(160, 5, f"LLM: {llm}", ln=True)
    
    pdf.ln(5)
    
    # System Prompt
    pdf.set_font("Times", 'B', 12)
    pdf.cell(160, 5, "System Prompt:", ln=False)
    pdf.set_font("Times", size=12)
    pdf.ln(0)  # Add a small line break
    pdf.multi_cell(160, 5, SYSTEM_PROMPT)
    pdf.ln(10)
    
    # Description
    pdf.set_font("Times", 'B', 12)
    pdf.cell(160, 5, "Description:", ln=True)
    pdf.set_font("Times", size=12)
    pdf.multi_cell(0, 25, "", border=1)  # Empty text box spanning the default page width
    pdf.ln(10)
    
    # Data
    pdf.set_font("Times", 'B', 14)
    pdf.cell(200, 10, "Data", ln=True)
    pdf.set_font("Times", size=12)
    
    col_width = 50
    row_height = 10
    spacing = 0
    
    for i in range(0, len(data_list), 3):
        for j in range(3):
            if i + j < len(data_list):
                row = data_list.iloc[i + j]
                pdf.cell(col_width - 15, row_height, f"Question {i + j + 1}", border=1, align='C')
                pdf.set_font("Times", 'B', 12)
                pdf.cell(15, row_height, f"{row['Average']}", border=1, align='C')
                pdf.set_font("Times", size=12)
            else:
                pdf.cell(col_width, row_height, "", border=1, align='C')
        pdf.ln(row_height + spacing)
    pdf.ln(10)
    
    # Force a page break before the graphs section
    pdf.add_page()
    
    # Graphs
    pdf.set_font("Times", 'B', 14)
    pdf.cell(160, 10, "Graphs", ln=True)
    for i, img in enumerate(images, 1):
        img_path = f"{DATA_FOLDER_PATH}/plots/temp_graph_{i}.png"
        img.savefig(img_path)
        img_width = 150
        x_position = (pdf.w - img_width) / 2  # Calculate the x position to center the image
        pdf.image(img_path, x=x_position, y=None, w=img_width)
        pdf.ln(10)
        os.remove(img_path)  # Delete the temporary graph image
    
    # Save PDF
    pdf_path = f"{DATA_FOLDER_PATH}/reports/{prefix}_evaluation_report.pdf"
    pdf.output(pdf_path)

    # Save the state after creating the report
    save_state(NAME, COUNTER)
    
    # Open the PDF
    subprocess.run(["open", pdf_path], check=True)