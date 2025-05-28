import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile
import plotly.io as pio

def get_csv_download_link(df, filename="data.csv", button_text="Download CSV"):
    """
    Generate a download link for a DataFrame as CSV
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to download
    filename : str
        The filename for the downloaded file
    button_text : str
        The text to display on the download button
    
    Returns:
    --------
    None (displays download button in the Streamlit app)
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def create_pdf_report(df, title, subtitle=None, description=None, figures=None):
    """
    Create a PDF report from a DataFrame and optional figures
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to include in the report
    title : str
        The title of the report
    subtitle : str, optional
        A subtitle for the report
    description : str, optional
        A description to include in the report
    figures : list, optional
        A list of matplotlib or plotly figures to include in the report
    
    Returns:
    --------
    BytesIO object containing the PDF
    """
    class PDF(FPDF):
        def header(self):
            # Logo
            # self.image('logo.png', 10, 8, 33)
            # Arial bold 15
            self.set_font('Arial', 'B', 15)
            # Move to the right
            self.cell(80)
            # Title
            self.cell(30, 10, title, 0, 0, 'C')
            # Line break
            self.ln(20)

        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Page number
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
            # Add date
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cell(-40, 10, current_date, 0, 0, 'R')

    # Create PDF object
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, 0, 1, 'C')
    
    # Add subtitle if provided
    if subtitle:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, subtitle, 0, 1, 'C')
    
    # Add description if provided
    if description:
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 10, description)
        pdf.ln(5)
    
    # Add figures note if provided
    if figures:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "Note: Interactive visualizations can be viewed in the app but are not included in the PDF export.", 0, 1, 'L')
        pdf.ln(5)
    
    # Add DataFrame
    pdf.set_font('Arial', 'B', 10)
    pdf.ln(10)
    pdf.cell(0, 10, "Data Table:", 0, 1, 'L')
    
    # Convert DataFrame to table in PDF
    pdf.set_font('Arial', 'B', 8)
    col_width = 180 / len(df.columns)
    
    # Print headers
    for col in df.columns:
        pdf.cell(col_width, 7, str(col)[:15], 1, 0, 'C')
    pdf.ln()
    
    # Print rows
    pdf.set_font('Arial', '', 8)
    for _, row in df.iloc[:50].iterrows():  # Limit to 50 rows for readability
        for val in row:
            val_str = str(val)
            if len(val_str) > 15:
                val_str = val_str[:12] + '...'
            pdf.cell(col_width, 6, val_str, 1, 0, 'L')
        pdf.ln()
    
    if len(df) > 50:
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f"Note: Only first 50 of {len(df)} rows are shown", 0, 1, 'L')
    
    # Return PDF as bytes
    pdf_output = io.BytesIO()
    pdf_output_path = pdf_output
    
    # FPDF requires a string for output name or a file object with write method
    # Use pdf_output.name for compatibility if available, otherwise use temp file
    if hasattr(pdf_output, 'name'):
        pdf.output(pdf_output.name)
        with open(pdf_output.name, 'rb') as f:
            pdf_output.write(f.read())
    else:
        # Alternative approach - write to temp file and then read back
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf.output(tmp_file.name)
            tmp_file_path = tmp_file.name
            
        with open(tmp_file_path, 'rb') as f:
            pdf_output.write(f.read())
            
        # Clean up temp file
        try:
            os.unlink(tmp_file_path)
        except:
            pass
    
    pdf_output.seek(0)
    return pdf_output

def get_pdf_download_link(pdf_bytes, filename="report.pdf", button_text="Download PDF"):
    """
    Generate a download link for a PDF report
    
    Parameters:
    -----------
    pdf_bytes : BytesIO
        BytesIO object containing the PDF data
    filename : str
        The filename for the downloaded file
    button_text : str
        The text to display on the download button
    
    Returns:
    --------
    None (displays download button in the Streamlit app)
    """
    b64 = base64.b64encode(pdf_bytes.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def add_export_section(df, title="Data Export", 
                       description="Export your data in various formats",
                       figures=None):
    """
    Add a data export section to the Streamlit app
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to export
    title : str
        The title of the export section
    description : str
        A description for the export section
    figures : list, optional
        A list of matplotlib or plotly figures to include in the PDF report
    
    Returns:
    --------
    None (displays export section in the Streamlit app)
    """
    if df is None or df.empty:
        st.warning("No data available to export. Please load or generate data first.")
        return
    
    st.subheader(title)
    st.write(description)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Export as CSV:")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_export_{timestamp}.csv"
        get_csv_download_link(df, filename=filename)
    
    with col2:
        st.write("Export as PDF Report:")
        # Create PDF report
        pdf_title = f"Data Report - {datetime.now().strftime('%Y-%m-%d')}"
        pdf_bytes = create_pdf_report(
            df=df,
            title=pdf_title,
            subtitle=title,
            description=description,
            figures=figures
        )
        # Provide download link
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"report_{timestamp}.pdf"
        get_pdf_download_link(pdf_bytes, filename=pdf_filename)
    
    st.info("Click on the links above to download your data in CSV or PDF format.")