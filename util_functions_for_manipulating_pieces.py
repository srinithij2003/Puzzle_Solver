from PIL import Image
import matplotlib.pyplot as plt
from puzzle_piece import Piece_of_Puzzle

def rotate_image_block(image_block, degrees):
    """
    Rotate a PIL Image block clockwise by 0, 90, 180, or 270 degrees.

    Args:
        image_block (PIL.Image.Image): The image to rotate.
        degrees (int): Rotation angle in degrees (clockwise). Must be 0, 90, 180, 270.

    Returns:
        PIL.Image.Image: A new rotated image.
    """
    if degrees not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees")
    
    # PIL rotates counter-clockwise, so negate for clockwise
    return image_block.rotate(-degrees, expand=True)


def rotate_image_and_edges(piece, degrees):
    """
    Returns a new Piece_of_Puzzle rotated clockwise by `degrees`.
    The original piece remains unchanged.
    """
    if degrees not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0, 90, 180, or 270")
    
    # Copy image and edges
    new_img = piece.image.rotate(-degrees, expand=True)
    
    order = ["top", "right", "bottom", "left"]
    steps = (degrees // 90) % 4
    new_edges = {order[(i + steps) % 4]: piece.edges[side] for i, side in enumerate(order)}
    
    return Piece_of_Puzzle(new_img, id=piece.id, edges=new_edges)


def combine_two_image_blocks(block_1, block_2):
    """
    Combine two image blocks along their longest dimension, only if widths or heights match.
    - If width matches -> horizontal merge (block1 left, block2 right)
    - If height matches -> vertical merge (block1 top, block2 bottom)
    - Otherwise, raise ValueError
    """
    w1, h1 = block_1.size
    w2, h2 = block_2.size

    # Case 1: Widths match -> horizontal merge
    if (w1 != w2) or (h1 != h2):
        raise ValueError(
            f"Cannot combine blocks: dimensions do not match.\n"
            f"Block1: {block_1.size}, Block2: {block_2.size}"
        )

    # If width longer than height, match on width
    if w1 > h1:
        # Vertical merge along height
        new_h = h1 + h2
        new_img = Image.new('RGB', (w1, new_h))
        new_img.paste(block_1, (0, 0))        # block1 on top
        new_img.paste(block_2, (0, h1))       # block2 below
        return new_img
    
    else:
        # Horizontal merge along width if width < or = height
        new_w = w1 + w2
        new_img = Image.new('RGB', (new_w, h1))
        new_img.paste(block_1, (0, 0))        # block1 on left
        new_img.paste(block_2, (w1, 0))       # block2 on right
        return new_img
    
def crop_seam(img, side, w, h, n):
    return img.crop((0, h-n, w, h+n)).rotate(90, expand=True) if side in ("top", "bottom") else img.crop((w-n, 0, w+n, h))

def combine_two_pieces(piece_a, 
                       piece_b, 
                       side_a, 
                       side_b, 
                       fontsize=10,
                       show_result = True,
                       return_edge_only=False, 
                       n=3):
    """
    Combine two Piece_of_Puzzle objects along the specified edges.
    Rotates piece_b to align side_b opposite of side_a.
    Displays the combined piece with edge labels.
    Optionally return only the meeting edge with n pixels from each side.
    """
    # Map (side_a, side_b) to clockwise rotation needed for piece_b
    rotation_matrix = {
        ("top", "top"): 180, ("top", "right"): 90, ("top", "bottom"): 0, ("top", "left"): 270,
        ("right", "top"): 270, ("right", "right"): 180, ("right", "bottom"): 90, ("right", "left"): 0,
        ("bottom", "top"): 0, ("bottom", "right"): 270, ("bottom", "bottom"): 180, ("bottom", "left"): 90,
        ("left", "top"): 90, ("left", "right"): 0, ("left", "bottom"): 270, ("left", "left"): 180,
    }

    degrees = rotation_matrix[(side_a, side_b)]
    piece_b_rotated = rotate_image_and_edges(piece_b, degrees)

    size = piece_a.size
    w, h = size, size

    # Combine images according to side_a
    if side_a == "top":
        new_img = Image.new('RGB', (w, h*2))
        new_img.paste(piece_b_rotated.image, (0, 0))
        new_img.paste(piece_a.image, (0, h))
        positions = [(piece_b_rotated, 0, 0), (piece_a, 0, h)]
    elif side_a == "bottom":
        new_img = Image.new('RGB', (w, h*2))
        new_img.paste(piece_a.image, (0, 0))
        new_img.paste(piece_b_rotated.image, (0, h))
        positions = [(piece_a, 0, 0), (piece_b_rotated, 0, h)]
    elif side_a == "left":
        new_img = Image.new('RGB', (w*2, h))
        new_img.paste(piece_b_rotated.image, (0, 0))
        new_img.paste(piece_a.image, (w, 0))
        positions = [(piece_b_rotated, 0, 0), (piece_a, w, 0)]
    elif side_a == "right":
        new_img = Image.new('RGB', (w*2, h))
        new_img.paste(piece_a.image, (0, 0))
        new_img.paste(piece_b_rotated.image, (w, 0))
        positions = [(piece_a, 0, 0), (piece_b_rotated, w, 0)]

    # Show combined image with labels
    if show_result:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(new_img)
        ax.axis('off')

        for piece, x_offset, y_offset in positions:
            w_p, h_p = piece.image.size
            if piece.edges["top"]:
                ax.text(x_offset + w_p/2, y_offset + 5, f"T:{piece.edges['top']}", ha='center', va='top', fontsize=fontsize, color='white')
            if piece.edges["right"]:
                ax.text(x_offset + w_p - 5, y_offset + h_p/2, f"R:{piece.edges['right']}", ha='right', va='center', fontsize=fontsize, color='white')
            if piece.edges["bottom"]:
                ax.text(x_offset + w_p/2, y_offset + h_p - 5, f"B:{piece.edges['bottom']}", ha='center', va='bottom', fontsize=fontsize, color='white')
            if piece.edges["left"]:
                ax.text(x_offset + 5, y_offset + h_p/2, f"L:{piece.edges['left']}", ha='left', va='center', fontsize=fontsize, color='white')
            ax.text(x_offset + w_p/2, y_offset + h_p/2 - 5, f"ID:{piece.id}", ha='center', va='top', fontsize=fontsize, color='black')

        plt.tight_layout()
        plt.show()

    if return_edge_only:
        return crop_seam(new_img, side_a, w, h, n)
    else:
        return new_img
