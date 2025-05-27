from PyPDF2 import PdfMerger

# Create a PdfMerger object
merger = PdfMerger()

# Paths to the PDF files you want to merge
pdf1 = '/Users/niyaz/Downloads/forms/2023W2.pdf'
pdf2 = '/Users/niyaz/Downloads/forms/2023W2_2.pdf'

# Append the PDFs to the merger object
merger.append(pdf1)
merger.append(pdf2)

# Write out the combined PDF
output = '/Users/niyaz/Downloads/forms/2023W2.pdf'
merger.write(output)
merger.close()

print(f"Combined PDF saved as {output}")
