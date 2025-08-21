

def partition_image(image_path, tile_size=(512, 512), overlap_ratio=0.2, output_dir="tiles"):
    """
    Partition a large satellite image into smaller overlapping tiles.
 
    Args:
        image_path (str): Path to the input satellite image
        tile_size (tuple): Size of each tile (width, height)
        overlap_ratio (float): Overlap ratio between adjacent tiles (0.0 to 0.5)
        output_dir (str): Directory to save the tiles
 
    Returns:
        int: Number of tiles created
    """
 
    # Read the image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
 
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
 
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
 
    # Calculate step size based on overlap
    tile_width, tile_height = tile_size
    step_x = int(tile_width * (1 - overlap_ratio))
    step_y = int(tile_height * (1 - overlap_ratio))
 
    print(f"Tile size: {tile_width}x{tile_height}")
    print(f"Step size: {step_x}x{step_y}")
    print(f"Overlap ratio: {overlap_ratio}")
 
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
 
    tile_count = 0
 
    # Generate tiles
    for y in range(0, height - tile_height + 1, step_y):
        for x in range(0, width - tile_width + 1, step_x):
            # Extract tile
            tile = image[y:y+tile_height, x:x+tile_width]
 
            # Generate filename with position information
            filename = f"tile_{tile_count:04d}_x{x}_y{y}.jpg"
            output_path = os.path.join(output_dir, filename)
 
            # Save tile
            cv2.imwrite(output_path, tile)
            tile_count += 1
 
            if tile_count % 50 == 0:
                print(f"Generated {tile_count} tiles...")
 
    print(f"✅ Successfully generated {tile_count} tiles in '{output_dir}'")
    return tile_count
