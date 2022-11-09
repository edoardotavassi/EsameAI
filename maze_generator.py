import random
from enum import Enum
from PIL import Image, ImageDraw, ImageFont


random.seed()


class State(Enum):
    EMPTY = 0
    START = 1
    PATH = 2
    WALL = 3
    TREASURE = 4


class Cell:
    def __init__(self, state):
        self.state = state
        if state == State.PATH:
            self.path_value = random.randint(1, 4)

    def change_state(self, state):
        self.state = state


class Maze(object):
    def __init__(
        self,
        dim,
        num_treasures,
    ):
        self.side_dim = dim
        self.matrix_dim = dim * dim
        self.cells = []
        for i in range(self.matrix_dim):
            if random.random() < 0.36:
                self.cells.append(Cell(State.WALL))
            else:
                self.cells.append((Cell(State.PATH)))
        count = 0
        while count < num_treasures:
            rnd = random.randint(0, self.matrix_dim - 1)
            if self.cells[rnd].state is not State.TREASURE:
                self.cells[rnd].change_state(State.TREASURE)
                count += 1

        while True:
            rnd = random.randint(0, self.matrix_dim - 1)
            if self.cells[rnd].state is not State.TREASURE:
                self.cells[rnd].change_state(State.START)
                break

    def visualize(self):
        OKGREEN = "\033[92m"
        WARNING = "\033[93m"
        OKCYAN = "\033[96m"
        CLEAN = "\033[0m"
        for i in range(self.matrix_dim):
            if i % (self.side_dim) == 0:
                print("")
            if self.cells[i].state == State.WALL:
                print(OKGREEN + "X" + CLEAN, end=" ")
            elif self.cells[i].state == State.PATH:
                print(OKCYAN, end="")
                print(self.cells[i].path_value, end=" ")
                print(CLEAN, end="")
            elif self.cells[i].state == State.TREASURE:
                print(WARNING + "T" + CLEAN, end=" ")
            else:
                print(WARNING + "S"+ CLEAN, end=" ")
        print("")


class Visualizer(object):
    def __init__(self, maze, media_filename):
        self.maze = maze
        self.dim = maze.side_dim
        self.media_filename = media_filename + ".jpg"

    def generate(self, height=1000, width=1000, padding=100):
        step_count = self.dim

        image = Image.new(mode="L", size=(height, width), color=255)

        # Draw some lines
        draw = ImageDraw.Draw(image)

        y_start = padding
        y_end = image.height - padding
        step_size = int((image.width - (padding * 2)) / step_count)
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", step_size)

        x = padding
        count = 0
        while count <= step_count:
            line = ((x, y_start), (x, y_end))
            draw.line(line, fill=0, width=3)
            x += step_size
            count += 1

        x_start = padding
        x_end = image.width - padding
        y = padding
        count = 0
        while count <= step_count:
            line = ((x_start, y), (x_end, y))
            draw.line(line, fill=0, width=3)
            y += step_size
            count += 1

        count_i = 0
        for y in range(padding + int(step_size / 2), image.width - padding, step_size):
            for x in range(
                padding + int(step_size / 2), image.height - padding, step_size
            ):
                if self.maze.cells[count_i].state == State.WALL:
                    draw.text((x, y), "X", fill=0, font=font, anchor="mm")
                elif self.maze.cells[count_i].state == State.PATH:
                    draw.text(
                        (x, y),
                        str(self.maze.cells[count_i].path_value),
                        fill=0,
                        font=font,
                        anchor="mm",
                    )
                elif self.maze.cells[count_i].state == State.TREASURE:
                    draw.text((x, y), "T", fill=0, font=font, anchor="mm")
                else:
                    draw.text((x, y), "S", fill=0, font=font, anchor="mm")
                count_i += 1

        del draw
        self.image = image

    def showImage(self):
        self.image.show()

    def saveImage(self, dir):
        self.image.save(dir + self.media_filename)
