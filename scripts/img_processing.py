#!/usr/bin/env python3
"""
Image Processing Script for Academic Webpages

This script provides a comprehensive set of utilities for processing images on academic websites:
1. Resize images to specific dimensions or aspect ratios
2. Create placeholder images with paper titles
3. Enhance real images with design elements (header bar, border, icon)
4. Compress images for web optimization
5. Generate thumbnails

Configuration can be specified via command line arguments or a config file.
"""

from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import sys
import textwrap
import argparse
import shutil
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "output_dir": "assets/img/publication_preview",
    "resize": {
        "enabled": False,
        "width": 800,
        "height": 600,
        "maintain_aspect_ratio": True,
        "high_quality": True
    },
    "placeholder": {
        "enabled": False,
        "width": 400,
        "height": 300,
        "bg_color": "#f8f9fa",
        "border_color": "#dee2e6",
        "header_color": "#007bff",
        "text_color": "#343a40",
        "header_text": "Research Paper"
    },
    "enhance": {
        "enabled": False,
        "add_header": True,
        "add_border": True,
        "add_icon": True,
        "header_height": 50,
        "header_color": "#007bff",
        "border_color": "#dee2e6",
        "icon_color": "#28a745"
    },
    "compress": {
        "enabled": False,
        "quality": 10,
        "optimize": False
    },
    "thumbnail": {
        "enabled": False,
        "width": 400,
        "height": 300,
        "suffix": "_thumb"
    },
    "backup": {
        "enabled": True,
        "dir": "originals"
    },
    "fonts": {
        "large": {
            "path": "/System/Library/Fonts/Helvetica.ttc", 
            "size": 24
        },
        "small": {
            "path": "/System/Library/Fonts/Helvetica.ttc", 
            "size": 16
        }
    }
}

def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_ext = os.path.splitext(config_path)[1].lower()
                if file_ext == '.yaml' or file_ext == '.yml':
                    user_config = yaml.safe_load(f)
                elif file_ext == '.json':
                    user_config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {file_ext}")
                    return config
                
                # Update config with user values (nested dict update)
                def update_dict(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                            d[k] = update_dict(d[k], v)
                        else:
                            d[k] = v
                    return d
                
                config = update_dict(config, user_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    return config

def load_fonts(config):
    """Load fonts from config or fallback to defaults."""
    fonts = {}
    
    for font_name, font_info in config["fonts"].items():
        try:
            fonts[font_name] = ImageFont.truetype(font_info["path"], font_info["size"])
            logger.debug(f"Loaded {font_name} font from {font_info['path']}")
        except Exception as e:
            logger.warning(f"Could not load {font_name} font: {e}")
            fonts[font_name] = ImageFont.load_default()
            logger.debug(f"Using default font for {font_name}")
    
    return fonts

def resize_image(img, config):
    """Resize image according to configuration."""
    width = config["resize"]["width"]
    height = config["resize"]["height"]
    maintain_aspect = config["resize"]["maintain_aspect_ratio"]
    high_quality = config["resize"]["high_quality"]
    
    if maintain_aspect:
        # Calculate new dimensions while preserving aspect ratio
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        
        # Calculate dimensions to maintain aspect ratio
        if width / height > aspect_ratio:
            new_width = int(height * aspect_ratio)
            new_height = height
        else:
            new_width = width
            new_height = int(width / aspect_ratio)
        
        # Use LANCZOS for high quality or BILINEAR for faster processing
        resample = Image.LANCZOS if high_quality else Image.BILINEAR
        img_resized = img.resize((new_width, new_height), resample=resample)
        
        # Create a new image with the target dimensions and paste the resized image
        new_img = Image.new("RGB", (width, height), color="white")
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        return new_img
    else:
        # Direct resize to specified dimensions
        resample = Image.LANCZOS if high_quality else Image.BILINEAR
        return img.resize((width, height), resample=resample)

def create_preview_image(title, filename, output_dir, config, fonts):
    """Create a simple preview image with the paper title."""
    cfg = config["placeholder"]
    width, height = cfg["width"], cfg["height"]
    
    # Create image with specified background color
    img = Image.new('RGB', (width, height), color=cfg["bg_color"])
    draw = ImageDraw.Draw(img)
    
    # Add a subtle border
    draw.rectangle([10, 10, width-10, height-10], outline=cfg["border_color"], width=2)
    
    # Add a colored header bar
    header_height = 60
    draw.rectangle([10, 10, width-10, header_height], fill=cfg["header_color"], outline=cfg["header_color"])
    
    # Add header text
    draw.text((20, 25), cfg["header_text"], fill='white', font=fonts["small"])
    
    # Wrap the title text
    wrapped_title = textwrap.fill(title, width=35)
    
    # Calculate text position to center it
    text_bbox = draw.textbbox((0, 0), wrapped_title, font=fonts["large"])
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (width - text_width) // 2
    y = 80 + (height - 140 - text_height) // 2
    
    # Add the title text
    draw.text((x, y), wrapped_title, fill=cfg["text_color"], font=fonts["large"], align='center')
    
    # Add a small academic icon or decoration
    draw.ellipse([width-60, height-60, width-20, height-20], fill='#28a745', outline='#28a745')
    draw.text((width-50, height-50), "📊", font=fonts["small"])
    
    # Save the image
    output_path = os.path.join(output_dir, filename)
    img.save(output_path)
    logger.info(f"Created: {output_path}")

def enhance_preview_image(input_path, output_path, config, fonts):
    """Enhance a real publication image with design elements."""
    cfg = config["enhance"]
    resize_cfg = config["resize"]
    
    # Target dimensions
    target_width, target_height = resize_cfg["width"], resize_cfg["height"]
    header_height = cfg["header_height"]
    
    # Load the real image
    try:
        img = Image.open(input_path)
        logger.info(f"Processing: {input_path}")
    except Exception as e:
        logger.error(f"Error opening {input_path}: {e}")
        return False
    
    # Resize the image using the resize function
    img = resize_image(img, config)
    
    # Create a new blank image with default background
    new_img = Image.new('RGB', (target_width, target_height), color='#f8f9fa')
    
    # Calculate position to center the resized image
    if cfg["add_header"]:
        x = (target_width - img.width) // 2
        y = header_height + (target_height - header_height - img.height) // 2  # Below header
    else:
        x = (target_width - img.width) // 2
        y = (target_height - img.height) // 2
    
    # Paste the resized image onto the new image
    new_img.paste(img, (x, y))
    
    # Now add the design elements
    draw = ImageDraw.Draw(new_img)
    
    # Add a subtle border around the entire image
    if cfg["add_border"]:
        draw.rectangle([0, 0, target_width-1, target_height-1], outline=cfg["border_color"], width=2)
    
    # Add a colored header bar
    if cfg["add_header"]:
        draw.rectangle([0, 0, target_width, header_height], fill=cfg["header_color"], outline=cfg["header_color"])
        # Add header text
        draw.text((20, (header_height-fonts["small"].size)//2), "Research Paper", fill='white', font=fonts["small"])
    
    # Add a small academic icon or decoration
    if cfg["add_icon"]:
        draw.ellipse([target_width-60, target_height-60, target_width-20, target_height-20], 
                    fill=cfg["icon_color"], outline=cfg["icon_color"])
        draw.text((target_width-50, target_height-50), "📊", font=fonts["small"])
    
    # Save the enhanced image
    new_img.save(output_path, quality=config["compress"]["quality"], optimize=config["compress"]["optimize"])
    logger.info(f"Enhanced image saved to: {output_path}")
    return True

def compress_image(input_path, output_path, config):
    """Compress an image for web optimization."""
    try:
        img = Image.open(input_path)
        
        # Check if image has transparency (PNG)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Convert to RGB first if it has transparency
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = background
        
        # Save with compression settings
        img.save(output_path, 
                quality=config["compress"]["quality"], 
                optimize=config["compress"]["optimize"])
        
        # Get file size reduction
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        reduction = (1 - compressed_size / original_size) * 100
        
        logger.info(f"Compressed {input_path} from {original_size/1024:.1f}KB to {compressed_size/1024:.1f}KB ({reduction:.1f}% reduction)")
        return True
    except Exception as e:
        logger.error(f"Error compressing {input_path}: {e}")
        return False

def create_thumbnail(input_path, output_path, config):
    """Create a thumbnail version of an image."""
    try:
        img = Image.open(input_path)
        width = config["thumbnail"]["width"]
        height = config["thumbnail"]["height"]
        
        # Create a copy and resize it for the thumbnail
        thumb = img.copy()
        thumb.thumbnail((width, height), Image.LANCZOS)
        
        # Save the thumbnail
        thumb.save(output_path, quality=config["compress"]["quality"], optimize=True)
        logger.info(f"Created thumbnail: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating thumbnail for {input_path}: {e}")
        return False

def process_images(input_dir, output_dir, config, fonts):
    """Process all images in the input directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create backup directory if enabled and doesn't exist
    if config["backup"]["enabled"]:
        backup_dir = os.path.join(output_dir, config["backup"]["dir"])
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"Backup directory: {backup_dir}")
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) and 
                  not f.startswith('.')]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_filename = image_file
        
        # Get output path (same as input if overwriting)
        if input_dir == output_dir:
            output_path = input_path
            # Backup original if enabled and overwriting originals
            if config["backup"]["enabled"]:
                backup_path = os.path.join(output_dir, config["backup"]["dir"], image_file)
                if not os.path.exists(backup_path):
                    shutil.copy2(input_path, backup_path)
                    logger.info(f"Backed up original to: {backup_path}")
        else:
            output_path = os.path.join(output_dir, output_filename)
        
        # Open the image
        try:
            img = Image.open(input_path)
        except Exception as e:
            logger.error(f"Error opening {input_path}: {e}")
            continue
        
        # Apply processing based on config
        if config["resize"]["enabled"]:
            img = resize_image(img, config)
            img.save(output_path)
            logger.info(f"Resized image saved to: {output_path}")
        
        if config["enhance"]["enabled"]:
            enhance_preview_image(input_path, output_path, config, fonts)
        
        if config["compress"]["enabled"] and not config["enhance"]["enabled"]:
            # Only compress separately if not already done during enhance
            compress_image(input_path, output_path, config)
        
        if config["thumbnail"]["enabled"]:
            filename, ext = os.path.splitext(output_filename)
            thumb_filename = f"{filename}{config['thumbnail']['suffix']}{ext}"
            thumb_path = os.path.join(output_dir, thumb_filename)
            create_thumbnail(output_path, thumb_path, config)
    
    logger.info("\nAll images have been processed!")
    if config["backup"]["enabled"] and input_dir == output_dir:
        logger.info(f"Original files have been backed up to: {os.path.join(output_dir, config['backup']['dir'])}")

def create_sample_config(output_path):
    """Create a sample configuration file."""
    try:
        with open(output_path, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Created sample configuration file at: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating config file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Academic webpage image processing utility')
    parser.add_argument('--config', type=str, help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--create-config', type=str, help='Create a sample configuration file at specified path')
    parser.add_argument('--input-dir', type=str, help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, help='Output directory for processed images')
    parser.add_argument('--resize', action='store_true', help='Enable image resizing')
    parser.add_argument('--width', type=int, help='Target image width')
    parser.add_argument('--height', type=int, help='Target image height')
    parser.add_argument('--placeholder', action='store_true', help='Create placeholder images')
    parser.add_argument('--enhance', action='store_true', help='Enhance images with design elements')
    parser.add_argument('--compress', action='store_true', help='Compress images for web optimization')
    parser.add_argument('--thumbnail', action='store_true', help='Create thumbnails')
    parser.add_argument('--backup', action='store_true', help='Create backup of original images')
    parser.add_argument('--no-backup', action='store_true', help='Disable backup of original images')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    # Paper info for placeholder creation
    parser.add_argument('--papers', type=str, help='JSON file with paper information for placeholders')
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config(args.create_config)
        return
    
    # Load configuration from file or use defaults
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = config["output_dir"]  # Default to output dir if not specified
    
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    if args.resize:
        config["resize"]["enabled"] = True
    if args.width:
        config["resize"]["width"] = args.width
    if args.height:
        config["resize"]["height"] = args.height
    if args.placeholder:
        config["placeholder"]["enabled"] = True
    if args.enhance:
        config["enhance"]["enabled"] = True
    if args.compress:
        config["compress"]["enabled"] = True
    if args.thumbnail:
        config["thumbnail"]["enabled"] = True
    if args.backup:
        config["backup"]["enabled"] = True
    if args.no_backup:
        config["backup"]["enabled"] = False
    
    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load fonts
    fonts = load_fonts(config)
    
    # Track what operations were performed
    operations = []
    
    # Create placeholder images if enabled
    if config["placeholder"]["enabled"]:
        operations.append("placeholder creation")
        if args.papers:
            try:
                # Load paper data from JSON
                with open(args.papers, 'r') as f:
                    papers = json.load(f)
                    
                logger.info("Creating placeholder preview images...")
                for paper in papers:
                    create_preview_image(paper["title"], paper["filename"], config["output_dir"], config, fonts)
                
            except Exception as e:
                logger.error(f"Error loading papers data: {e}")
                logger.info("Using sample paper data instead...")
                # Sample paper data
                papers = [
                    {"title": "The ADAIO System: AI Teacher Responses in Educational Dialogues", "filename": "adaio_system.png"},
                    {"title": "The ART of Conversation: Measuring Phonetic Convergence with Siamese RNN", "filename": "entrainment_siamese.png"},
                    {"title": "ART Corpus: Speech Entrainment and Imitation Dataset", "filename": "art_corpus.png"},
                    {"title": "Language Proficiency and F0 Entrainment in L2 English", "filename": "f0_entrainment.png"},
                    {"title": "Breathing Features and Speech Perception in COVID-19 Patients", "filename": "covid_breathing.png"},
                ]
                
                for title, filename in papers:
                    create_preview_image(title, filename, config["output_dir"], config, fonts)
        else:
            logger.warning("No papers data provided for placeholder creation")
    
    # Process existing images
    if any([config["resize"]["enabled"], config["enhance"]["enabled"], 
            config["compress"]["enabled"], config["thumbnail"]["enabled"]]):
        
        if config["resize"]["enabled"]:
            operations.append("resizing")
        if config["enhance"]["enabled"]:
            operations.append("enhancement")
        if config["compress"]["enabled"]:
            operations.append("compression")
        if config["thumbnail"]["enabled"]:
            operations.append("thumbnail creation")
        
        process_images(input_dir, config["output_dir"], config, fonts)
    
    # Summary
    if operations:
        logger.info(f"\nCompleted operations: {', '.join(operations)}")
        logger.info(f"Output directory: {config['output_dir']}")
    else:
        logger.warning("No operations were performed. Use flags to enable image processing options.")
        logger.info("Run with --help for usage information or --create-config to generate a config file.")

if __name__ == "__main__":
    main()
