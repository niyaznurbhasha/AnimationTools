import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Set the directory containing the HTML files
directory = '/mnt/c/Users/niyaz/Documents/CraftsMan/alexander_2/'  # Make sure this is the correct path

# Setting up Chrome options for headless mode
options = Options()
options.headless = True
options.add_argument('--window-size=1920,1080')  # Adjust as needed

# Setting up Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Process each file in the directory
for filename in os.listdir(directory):
    print(filename)
    if filename.endswith('.webp'):  # Ensure it's an HTML file
        # Construct the full file path
        print(filename)
        file_path = os.path.join(directory, filename)
        file_url = 'file:///' + file_path.replace(os.sep, '/')
        driver.get(file_url)

        # Convert filename from .html to .png
        png_filename = filename[:-5] + '.png'
        output_path = os.path.join(directory, png_filename)

        # Take screenshot
        driver.save_screenshot(output_path)
        print(f'Screenshot saved for {filename} at {output_path}')

driver.quit()
print('All screenshots have been saved.')
