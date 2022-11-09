import maze_generator as mg
import computer_vision_maze as cvm
import treasure_maze
import os

print("Vuoi generare un labirinto casuale o caricare un labirinto da file?")
print("1. Genera labirinto casuale")
print("2. Carica labirinto da file")
maze_type = input("Scelta: ")
if maze_type == "1":
    if os.path.exists("./img/maze.jpg"):
        os.remove("./img/maze.jpg")
        
    print("Dimensione del labirinto?")
    choice = input("Scelta: ")
    maze = mg.Maze(int(choice), int(choice))
    maze.visualize()
    visual = mg.Visualizer(maze, "maze")
    visual.generate()
    #visual.showImage()
    visual.saveImage("img/")
    analyzer = cvm.ImageAnalysis("./img/maze.jpg")
    map = analyzer.digital_maze_detection()
    print(map)
    
elif maze_type == "2":
    exit()
else:
    exit()
