#!/usr/bin/env python3
"""
Script to create and enhance preview images for publications.
This script can:
1. Create placeholder images with paper titles
2. Enhance real images with design elements (header bar, border, icon)
"""

from PIL import Image, ImageDraw, ImageFont
import os
import textwrap
import argparse
import shutil

def create_preview_image(title, filename, output_dir):
    """Create a simple preview image with the paper title."""
    # Image dimensions
    width, height = 400, 300
    
    # Create image with a nice gradient background
    img = Image.new('RGB', (width, height), color='#f8f9fa')
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        # Try different font sizes
        font_large = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 24)
        font_small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add a subtle border
    draw.rectangle([10, 10, width-10, height-10], outline='#dee2e6', width=2)
    
    # Add a colored header bar
    draw.rectangle([10, 10, width-10, 60], fill='#007bff', outline='#007bff')
    
    # Add "Research Paper" text in header
    draw.text((20, 25), "Research Paper", fill='white', font=font_small)
    
    # Wrap the title text
    wrapped_title = textwrap.fill(title, width=35)
    
    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), wrapped_title, font=font_large)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (width - text_width) // 2
    y = 80 + (height - 140 - text_height) // 2
    
    # Add the title text
    draw.text((x, y), wrapped_title, fill='#343a40', font=font_large, align='center')
    
    # Add a small academic icon or decoration
    draw.ellipse([width-60, height-60, width-20, height-20], fill='#28a745', outline='#28a745')
    draw.text((width-50, height-50), "📊", font=font_small)
    
    # Save the image
    output_path = os.path.join(output_dir, filename)
    img.save(output_path)
    print(f"Created: {output_path}")

def enhance_preview_image(input_path, output_path):
    """Enhance a real publication image with design elements."""
    # Target dimensions
    target_width, target_height = 400, 300
    
    # Load the real image
    try:
        img = Image.open(input_path)
        print(f"Processing: {input_path}")
    except Exception as e:
        print(f"Error opening {input_path}: {e}")
        return False
    
    # Resize the image to fit within dimensions while preserving aspect ratio
    img.thumbnail((target_width, target_height - 50))  # Leave room for header
    
    # Create a new blank image with white background
    new_img = Image.new('RGB', (target_width, target_height), color='#f8f9fa')
    
    # Calculate position to center the resized image
    x = (target_width - img.width) // 2
    y = 50 + (target_height - 50 - img.height) // 2  # Below header
    
    # Paste the resized image onto the new image
    new_img.paste(img, (x, y))
    
    # Now add the design elements
    draw = ImageDraw.Draw(new_img)
    
    # Try to use a nice font, fallback to default
    try:
        font_small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
        print("Using custom font for header and icon")
    except:
        font_small = ImageFont.load_default()
        print("Using default font for header and icon")
    
    # Add a subtle border around the entire image
    draw.rectangle([0, 0, target_width-1, target_height-1], outline='#dee2e6', width=2)
    
    # Add a colored header bar
    draw.rectangle([0, 0, target_width, 50], fill='#007bff', outline='#007bff')
    
    # Add "Research Paper" text in header
    draw.text((20, 17), "Research Paper", fill='white', font=font_small)
    
    # Add a small academic icon or decoration
    draw.ellipse([target_width-60, target_height-60, target_width-20, target_height-20], 
                 fill='#28a745', outline='#28a745')
    draw.text((target_width-50, target_height-50), "📊", font=font_small)
    
    # Save the enhanced image
    new_img.save(output_path)
    print(f"Enhanced image saved to: {output_path}")
    return True

def process_real_images(input_dir, backup=False):
    """Process all real images in the input directory"""
    # Create backup directory if requested
    backup_dir = os.path.join(input_dir, "originals")
    if backup and not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) and 
                  not f.startswith('.')]
    
    print(f"Found {len(image_files)} images to enhance")
    
    # Process each image
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        
        # Backup original if requested
        if backup:
            backup_path = os.path.join(backup_dir, image_file)
            shutil.copy2(input_path, backup_path)
            print(f"Backed up original to: {backup_path}")
        
        output_path = os.path.join(input_dir, image_file)  # Overwrite existing file
        enhance_preview_image(input_path, output_path)
    
    print("\nAll images have been enhanced!")
    if backup:
        print(f"Original files have been backed up to: {backup_dir}")
    else:
        print("Note: Original files have been overwritten with enhanced versions.")

def main():
    parser = argparse.ArgumentParser(description='Create or enhance publication preview images')
    parser.add_argument('--mode', type=str, choices=['create', 'enhance', 'both'], default='create',
                      help='Mode: create placeholder images, enhance real images, or both')
    parser.add_argument('--backup', action='store_true', 
                      help='Create backup of original images before enhancing')
    parser.add_argument('--output-dir', type=str, default='assets/img/publication_preview',
                      help='Directory for output images')
    
    args = parser.parse_args()
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if args.mode in ['create', 'both']:
        # Paper information
        papers = [
            ("The ADAIO System: AI Teacher Responses in Educational Dialogues", "adaio_system.png"),
            ("The ART of Conversation: Measuring Phonetic Convergence with Siamese RNN", "entrainment_siamese.png"),
            ("ART Corpus: Speech Entrainment and Imitation Dataset", "art_corpus.png"),
            ("Language Proficiency and F0 Entrainment in L2 English", "f0_entrainment.png"),
            ("Breathing Features and Speech Perception in COVID-19 Patients", "covid_breathing.png"),
        ]
        
        # Create placeholder preview images
        print("Creating placeholder preview images...")
        for title, filename in papers:
            create_preview_image(title, filename, output_dir)
        
        print(f"\nAll placeholder preview images created in {output_dir}/")
    
    if args.mode in ['enhance', 'both']:
        # Enhance real images
        print("\nEnhancing real preview images...")
        process_real_images(output_dir, backup=args.backup)

if __name__ == "__main__":
    main()
