from PIL import Image
import matplotlib.pyplot as plt
import itertools

class IDGenerator:
    """Simple global ID generator."""
    def __init__(self):
        self._counter = itertools.count()
    def get_new_id(self):
        return next(self._counter)

# Global ID generators
edge_id_gen = IDGenerator()
piece_id_gen = IDGenerator()

# For resetting generator back to 0
def reset_id_generators():
    global edge_id_gen, piece_id_gen
    edge_id_gen = IDGenerator()
    piece_id_gen = IDGenerator()

class Piece_of_Puzzle:
    def __init__(self, image, row=0, col=0, id=None, edges=None):
        """
        image: PIL.Image
        row, col: optional position info
        id: optional piece ID
        edges: optional dict of edge IDs
        """
        self.image = image
        self.row = row
        self.col = col
        self.size = image.size[0]  # assume square

        # Assign unique piece ID
        self.id = piece_id_gen.get_new_id() if id is None else id

        # Assign unique edges
        if edges is None:
            self.edges = {
                "top": f"E{edge_id_gen.get_new_id()}",
                "right": f"E{edge_id_gen.get_new_id()}",
                "bottom": f"E{edge_id_gen.get_new_id()}",
                "left": f"E{edge_id_gen.get_new_id()}"
            }
        else:
            self.edges = edges

    def get_edge_id(self, side):
        return self.edges.get(side)

    def rotate(self, degrees):
        """
        Rotate the piece clockwise by 0, 90, 180, or 270 degrees.
        Rotates both image and edges in-place.
        """
        if degrees not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180, or 270")

        if degrees == 0:
            return

        # Rotate image (PIL rotates counter-clockwise)
        self.image = self.image.rotate(-degrees, expand=True)

        # Rotate edges clockwise
        order = ["top", "right", "bottom", "left"]
        steps = (degrees // 90) % 4
        self.edges = {order[(i + steps) % 4]: self.edges[side] for i, side in enumerate(order)}

    def show_with_labels(self, ax=None, fontsize=10, color="white"):
        """
        Display the image with edge labels using Matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        ax.imshow(self.image)
        ax.axis("off")
        w, h = self.image.size

        # Draw edge labels
        if self.edges["top"]:
            ax.text(w / 2, 5, f"T:{self.edges['top']}", ha='center', va='top', fontsize=fontsize, color=color)
        if self.edges["right"]:
            ax.text(w - 5, h / 2, f"R:{self.edges['right']}", ha='right', va='center', fontsize=fontsize, color=color)
        if self.edges["bottom"]:
            ax.text(w / 2, h - 5, f"B:{self.edges['bottom']}", ha='center', va='bottom', fontsize=fontsize, color=color)
        if self.edges["left"]:
            ax.text(5, h / 2, f"L:{self.edges['left']}", ha='left', va='center', fontsize=fontsize, color=color)

        # Display piece ID
        ax.text(w / 2, h / 2 - 5, f"ID:{self.id}", ha='center', va='top', fontsize=fontsize, color=color)

        if ax is None:
            plt.show()

    def get_edge_strip(self, edge_id, n):
        """
        Return an image strip of thickness n pixels from the edge
        identified by edge_id.
        """

        # Find which side matches the edge_id
        side = None
        for k, v in self.edges.items():
            if v == edge_id:
                side = k
                break

        if side is None:
            raise ValueError(f"Edge ID {edge_id} not found in this piece")

        w, h = self.image.size

        if side == "top":
            box = (0, 0, w, n)
        elif side == "bottom":
            box = (0, h - n, w, h)
        elif side == "left":
            box = (0, 0, n, h)
        elif side == "right":
            box = (w - n, 0, w, h)
        else:
            raise ValueError(f"Unknown edge side: {side}")

        return self.image.crop(box)
    
    def get_edge_strips(self, edge_id, n):
        """
        Return strips starting from the given edge, moving inward.
        Strip sizes:
        - Left/Right edges: S x n
        - Top/Bottom edges: n x S
        Any leftover that cannot form a full strip is discarded.
        """
        # Find which side matches the edge_id
        side = None
        for k, v in self.edges.items():
            if v == edge_id:
                side = k
                break

        if side is None:
            raise ValueError(f"Edge ID {edge_id} not found in this piece")

        S = self.image.size[0]  # assume square
        num_strips = S // n
        strips = []

        if side == "left":
            for i in range(num_strips):
                x0 = 0
                y0 = i * n
                box = (x0, y0, x0 + S, y0 + n)  # width S, height n
                strips.append(self.image.crop(box))

        elif side == "right":
            for i in range(num_strips):
                x0 = 0
                y0 = i * n
                box = (x0, y0, x0 + S, y0 + n)  # width S, height n
                strips.append(self.image.crop(box))

        elif side == "top":
            for i in range(num_strips):
                x0 = i * n
                y0 = 0
                box = (x0, y0, x0 + n, y0 + S)  # width n, height S
                strips.append(self.image.crop(box))

        elif side == "bottom":
            for i in range(num_strips):
                x0 = i * n
                y0 = 0
                box = (x0, y0, x0 + n, y0 + S)  # width n, height S
                strips.append(self.image.crop(box))

        else:
            raise ValueError(f"Unknown edge side: {side}")

        return strips

    def __repr__(self):
        return f"<Piece_of_Puzzle row={self.row}, col={self.col}, edges={self.edges}>"
