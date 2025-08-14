# Synthetic toy PDF generator and corresponding ground truth .json
# for unit testing vision-LLM / OCR / PDF parsers
# 
# Execute with:
#   python synthesis.py --seed 1
#
# m.mieskolainen@imperial.ac.uk, 2025

import os
import uuid
import json
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from faker import Faker

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, PageBreak, Flowable)
from reportlab.lib.styles import getSampleStyleSheet

# Initialize Faker for realistic text generation.
fake = Faker()

# Global dictionaries to store computed bounding boxes and metadata chunks keyed by metadata id.
METADATA_BBOX = {}
METADATA_MAP = {}

# --- Custom Wrapper to Record Page Numbers per Flowable ---
class MetaFlowable(Flowable):
    def __init__(self, flowable):
        super().__init__()
        self.flowable = flowable
        # Copy the _metadata_id from the wrapped flowable (if it exists)
        self._metadata_id = getattr(flowable, "_metadata_id", None)
    
    def wrap(self, availWidth, availHeight):
        return self.flowable.wrap(availWidth, availHeight)
    
    def draw(self):
        self.flowable.canv = self.canv  # Pass the canvas reference.
        self.flowable.draw()
        # Immediately update the page number in the corresponding metadata.
        if self._metadata_id and self._metadata_id in METADATA_MAP:
            METADATA_MAP[self._metadata_id]["grounding"][0]["page"] = self.canv.getPageNumber() - 1  # 0-indexed page numbers

def meta(flowable):
    """Helper: wrap a flowable in a MetaFlowable."""
    return MetaFlowable(flowable)

# --- Custom DocTemplate ---
class MyDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        # If the flowable is wrapped in MetaFlowable, use its _metadata_id.
        metadata_id = getattr(flowable, "_metadata_id", None)
        if metadata_id:
            page_width, page_height = self.pagesize  # For letter: (612, 792)
            w, h = flowable.wrap(self.width, self.height)
            x = self.leftMargin
            # Use _y (which is updated by ReportLab) as current vertical position.
            y = self.canv._y  
            norm_l = x / page_width
            norm_r = (x + w) / page_width
            norm_b = (page_height - y) / page_height
            norm_t = (page_height - (y + h)) / page_height
            METADATA_BBOX[metadata_id] = {
                "l": round(norm_l, 3),
                "t": round(norm_t, 3),
                "r": round(norm_r, 3),
                "b": round(norm_b, 3)
            }
            # Also update the page number in the metadata.
            if metadata_id in METADATA_MAP:
                METADATA_MAP[metadata_id]["grounding"][0]["page"] = self.canv.getPageNumber() - 1  # 0-indexed page numbers

def new_chunk(text, chunk_type, page, chunk_index, extra_metadata=None):
    """
    Generate a metadata chunk with a unique ID.
    The 'page' field is initially set to the passed value but will be updated later.
    Extra metadata (such as table data or plot arrays) is merged into the chunk.
    """
    chunk_id = str(uuid.uuid4())
    chunk = {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "text": text,
        "grounding": [{
            "box": None,  # To be updated after layout.
            "page": page
        }]
    }
    # Add extra metadata directly to the chunk instead of merging
    if extra_metadata:
        for key, value in extra_metadata.items():
            chunk[key] = value
    
    METADATA_MAP[chunk_id] = chunk  # Save for later updating.
    return chunk_id, chunk

def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    canvas.drawRightString(doc.pagesize[0] - 72, 30, f"Page {page_num}")

def save_figure(array, filename, title="2D Visualization"):
    plt.figure()
    plt.imshow(array, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_1d_plot(x, y, title, filename, xlabel="X-axis", ylabel="Y-axis"):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_scatter_plot(x, y, filename, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_section_paragraphs(n):
    """Generate n realistic paragraphs using Faker."""
    return [fake.paragraph(nb_sentences=random.randint(8, 12)) for _ in range(n)]

def generate_section(section_name, min_paras, max_paras, include_figures, include_tables, page, start_chunk_index, styles):
    """
    Generate a section's flowables with a heading, paragraphs, and optionally figures/tables.
    Returns the list of flowables and the updated chunk index.
    """
    flowables = []
    chunk_index = start_chunk_index

    # Add section heading.
    heading = section_name
    p = Paragraph(heading, styles["Heading1"])
    cid, chunk = new_chunk(heading, "heading", page, chunk_index)
    p._metadata_id = cid
    flowables.append(Spacer(1, 12))
    flowables.append(meta(p))
    chunk_index += 1

    # Generate paragraphs.
    num_paras = random.randint(min_paras, max_paras)
    paragraphs = generate_section_paragraphs(num_paras)
    content_items = [("paragraph", para) for para in paragraphs]

    # Determine number of figures and tables.
    num_figures = random.randint(0, 2) if include_figures else 0
    num_tables = random.randint(0, 1) if include_tables else 0

    # Append figures.
    figure_types = ["2D", "scatter", "sinusoidal", "quadratic"]
    for _ in range(num_figures):
        content_items.append(("figure", random.choice(figure_types)))
    # Append tables.
    for _ in range(num_tables):
        content_items.append(("table", None))

    random.shuffle(content_items)

    for item in content_items:
        if item[0] == "paragraph":
            para = item[1]
            p = Paragraph(para, styles["Normal"])
            cid, chunk = new_chunk(para, "text", page, chunk_index)
            p._metadata_id = cid
            flowables.append(Spacer(1, 12))
            flowables.append(meta(p))
            chunk_index += 1
        elif item[0] == "figure":
            fig_type = item[1]
            if fig_type == "2D":
                array = np.random.rand(10, 10)
                fname = f"temp_2D_{page}_{chunk_index}.png"
                save_figure(array, fname, title="2D Visualization")
                img = Image(fname, width=400, height=300)
                extra = {"array": array.tolist()}
                cid, chunk = new_chunk("Figure: 2D Visualization", "figure", page, chunk_index, extra_metadata=extra)
                img._metadata_id = cid
                flowables.append(Spacer(1, 12))
                flowables.append(meta(img))
                caption = Paragraph("Figure: 2D Visualization", styles["Normal"])
                flowables.append(Spacer(1, 6))
                flowables.append(meta(caption))
                chunk_index += 1
            elif fig_type == "scatter":
                x_scatter = np.random.rand(50)
                y_scatter = np.random.rand(50)
                fname = f"temp_scatter_{page}_{chunk_index}.png"
                save_scatter_plot(x_scatter, y_scatter, fname, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis")
                img = Image(fname, width=400, height=300)
                extra = {"scatter_data": {"x": x_scatter.tolist(), "y": y_scatter.tolist()}}
                cid, chunk = new_chunk("Figure: Scatter Plot", "figure", page, chunk_index, extra_metadata=extra)
                img._metadata_id = cid
                flowables.append(Spacer(1, 12))
                flowables.append(meta(img))
                caption = Paragraph("Figure: Scatter Plot", styles["Normal"])
                flowables.append(Spacer(1, 6))
                flowables.append(meta(caption))
                chunk_index += 1
            elif fig_type == "sinusoidal":
                x = np.linspace(0, 10, 100)
                y = np.sin(x) + random.uniform(-0.1, 0.1)
                fname = f"temp_sinusoidal_{page}_{chunk_index}.png"
                save_1d_plot(x, y, title="Sinusoidal Plot", filename=fname, xlabel="Time (s)", ylabel="Temperature (°C)")
                img = Image(fname, width=400, height=300)
                extra = {"plot_data": {"x": x.tolist(), "y": y.tolist()}}
                cid, chunk = new_chunk("Figure: Sinusoidal Plot", "figure", page, chunk_index, extra_metadata=extra)
                img._metadata_id = cid
                flowables.append(Spacer(1, 12))
                flowables.append(meta(img))
                caption = Paragraph("Figure: Sinusoidal Plot", styles["Normal"])
                flowables.append(Spacer(1, 6))
                flowables.append(meta(caption))
                chunk_index += 1
            elif fig_type == "quadratic":
                x = np.linspace(0, 10, 100)
                y = x**2 + random.uniform(-5, 5)
                fname = f"temp_quadratic_{page}_{chunk_index}.png"
                save_1d_plot(x, y, title="Quadratic Plot", filename=fname, xlabel="Time (s)", ylabel="Velocity (m/s)")
                img = Image(fname, width=400, height=300)
                extra = {"plot_data": {"x": x.tolist(), "y": y.tolist()}}
                cid, chunk = new_chunk("Figure: Quadratic Plot", "figure", page, chunk_index, extra_metadata=extra)
                img._metadata_id = cid
                flowables.append(Spacer(1, 12))
                flowables.append(meta(img))
                caption = Paragraph("Figure: Quadratic Plot", styles["Normal"])
                flowables.append(Spacer(1, 6))
                flowables.append(meta(caption))
                chunk_index += 1
        elif item[0] == "table":
            rows = random.randint(3, 6)
            table_data = [["Header", "Value"]]
            for _ in range(rows):
                table_data.append([fake.word(), f"{random.uniform(0, 100):.2f}"])
            t = Table(table_data)
            extra = {"table_data": table_data}
            cid, chunk = new_chunk("Table: Data Table", "table", page, chunk_index, extra_metadata=extra)
            t._metadata_id = cid
            flowables.append(Spacer(1, 12))
            flowables.append(meta(t))
            caption = Paragraph("Table: Data Table", styles["Normal"])
            flowables.append(Spacer(1, 6))
            flowables.append(meta(caption))
            chunk_index += 1
    return flowables, chunk_index

def generate_references():
    """Generate a random reference list."""
    num_refs = random.randint(2, 5)
    refs = []
    for i in range(num_refs):
        author = fake.name()
        title = fake.sentence(nb_words=6)
        journal = fake.company()
        year = random.randint(1990, 2023)
        ref = f"[{i+1}] {author}. {title}. {journal}, {year}."
        refs.append(ref)
    return "<br/>".join(refs)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a variable synthetic scientific PDF with metadata."
    )
    parser.add_argument("--seed", type=int, default=int(time.time()),
                        help="Random seed (default: current time)")
    parser.add_argument("--output", type=str, default="output/",
                        help="Output folder (default: 'output/')")
    args = parser.parse_args()
    seed = args.seed
    output_folder = args.output
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_filename = os.path.join(output_folder, f"synthetic_document_{seed}.pdf")
    metadata_filename = os.path.join(output_folder, f"synthetic_metadata_{seed}.json")
    
    doc = MyDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    chunk_index = 0
    current_page = 0
    
    # TITLE PAGE: Random paper title, authors, abstract.
    title_text = fake.catch_phrase()
    p = Paragraph(title_text, styles["Title"])
    cid, chunk = new_chunk(title_text, "marginalia", current_page, chunk_index)
    p._metadata_id = cid
    story.append(meta(p))
    chunk_index += 1
    
    authors_text = fake.name() + ", " + fake.name() + ", " + fake.name() + "<br/>" + "Affiliations: " + fake.company()
    p = Paragraph(authors_text, styles["Normal"])
    cid, chunk = new_chunk(authors_text, "text", current_page, chunk_index)
    p._metadata_id = cid
    story.append(Spacer(1,12))
    story.append(meta(p))
    chunk_index += 1
    
    abstract_heading = "Abstract"
    p = Paragraph(abstract_heading, styles["Heading1"])
    cid, chunk = new_chunk(abstract_heading, "heading", current_page, chunk_index)
    p._metadata_id = cid
    story.append(Spacer(1,12))
    story.append(meta(p))
    chunk_index += 1
    
    for para in generate_section_paragraphs(random.randint(2,4)):
        p = Paragraph(para, styles["Normal"])
        cid, chunk = new_chunk(para, "text", current_page, chunk_index)
        p._metadata_id = cid
        story.append(Spacer(1,12))
        story.append(meta(p))
        chunk_index += 1
    
    story.append(PageBreak())
    current_page += 1
    chunk_index = 0
    
    # BASE SECTIONS.
    base_sections = [
        {"name": "Introduction", "min_paras": 3, "max_paras": 6, "include_figures": random.choice([True, False]), "include_tables": random.choice([True, False])},
        {"name": "Methods", "min_paras": 3, "max_paras": 6, "include_figures": True, "include_tables": True},
        {"name": "Results", "min_paras": 3, "max_paras": 6, "include_figures": True, "include_tables": True},
        {"name": "Discussion", "min_paras": 3, "max_paras": 6, "include_figures": random.choice([True, False]), "include_tables": random.choice([True, False])},
        {"name": "Conclusion", "min_paras": 2, "max_paras": 4, "include_figures": False, "include_tables": False},
    ]
    if random.random() < 0.5:
        base_sections.insert(-1, {"name": "Future Work", "min_paras": 2, "max_paras": 4, "include_figures": random.choice([True, False]), "include_tables": False})
    
    for sec in base_sections:
        sec_flowables, chunk_index = generate_section(sec["name"], sec["min_paras"], sec["max_paras"], sec["include_figures"], sec["include_tables"], current_page, chunk_index, styles)
        story.extend(sec_flowables)
        story.append(PageBreak())
        current_page += 1
        chunk_index = 0
    
    # REFERENCES SECTION.
    ref_heading = "References"
    p = Paragraph(ref_heading, styles["Heading1"])
    cid, chunk = new_chunk(ref_heading, "heading", current_page, chunk_index)
    p._metadata_id = cid
    story.append(Spacer(1, 24))
    story.append(meta(p))
    chunk_index += 1
    references_text = generate_references()
    p = Paragraph(references_text, styles["Normal"])
    cid, chunk = new_chunk(references_text, "text", current_page, chunk_index)
    p._metadata_id = cid
    story.append(Spacer(1, 12))
    story.append(meta(p))
    chunk_index += 1
    
    # Use multiBuild so that all flowables are drawn and page numbers are finalized.
    doc.multiBuild(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    
    # Get updated metadata chunks
    metadata_chunks = list(METADATA_MAP.values())
    
    # Update metadata chunks with computed bounding boxes.
    for chunk in metadata_chunks:
        metadata_id = chunk["chunk_id"]
        if metadata_id in METADATA_BBOX:
            chunk["grounding"][0]["box"] = METADATA_BBOX[metadata_id]
        else:
            chunk["grounding"][0]["box"] = {"l": None, "t": None, "r": None, "b": None}
    
    # Generate markdown from metadata chunks
    # Sort chunks by page number and position for proper sequential extraction
    sorted_chunks = sorted(metadata_chunks, 
                           key=lambda c: (c["grounding"][0]["page"], 
                                          c["grounding"][0]["box"]["t"] if c["grounding"][0]["box"]["t"] is not None else 1))
    
    markdown_lines = []
    for chunk in sorted_chunks:
        if chunk["chunk_type"] in ["marginalia", "heading", "text"]:
            markdown_lines.append(chunk["text"])
    
    full_markdown = "\n\n".join(markdown_lines)
    final_metadata = {
        "seed": seed,
        "data": {
            "markdown": full_markdown,
            "chunks": metadata_chunks
        }
    }
    
    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2)
    
    # Clean up temporary image files.
    temp_files = [fname for fname in os.listdir('.') if fname.startswith("temp_") or fname.startswith("temp_scatter_") 
                  or fname.startswith("temp_sinusoidal_") or fname.startswith("temp_quadratic_")]
    for fname in temp_files:
        os.remove(fname)
    
    print(f"PDF generated as '{pdf_filename}'")
    print(f"Metadata generated as '{metadata_filename}'")
    print(f"Random seed used: {seed}")

if __name__ == "__main__":
    main()

