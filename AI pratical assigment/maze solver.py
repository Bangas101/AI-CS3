import random
import turtle

#CONFIGURATION
# Set the maze dimensions. Larger numbers will result in a more complex maze and a longer generation time.
MAZE_WIDTH = 5
MAZE_HEIGHT = 5

# Set the turtle drawing speed. 0 is the fastest.
TURTLE_SPEED = 0

# Set the start and end colors for the path.
PATH_COLOR = "red"

#HELPER FUNCTIONS

def list_sum(input_list):
    """Calculates the sum of all elements in a 2D list."""
    total_sum = 0
    for row in input_list:
        total_sum += sum(row)
    return total_sum

def get_opposite_node(direction, current_node):
    """
    Calculates the coordinates of the cell adjacent to the current cell in a given direction.
    Directions are: 0=left, 1=top, 2=right, 3=bottom.
    """
    x, y = current_node
    if direction == 0:
        return [x - 1, y]
    elif direction == 1:
        return [x, y - 1]
    elif direction == 2:
        return [x + 1, y]
    else:  # direction == 3
        return [x, y + 1]

#MAZE GENERATION
def generate_maze(width, height):
    """
    Generates a maze using the recursive backtracking algorithm.
    Returns a 2D list representing the walls of the maze.
    Walls are represented as [left, top, right, bottom] with 1=wall, 0=no wall.
    """
    walls = [[[1, 1, 1, 1] for _ in range(width)] for _ in range(height)]
    visited = [[0 for _ in range(width)] for _ in range(height)]

    # Start at a random cell
    start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)

    path_stack = [(start_x, start_y)]
    visited[start_y][start_x] = 1

    while list_sum(visited) < width * height:
        current_x, current_y = path_stack[-1]

        # Find unvisited neighbors
        unvisited_neighbors = []
        if current_x > 0 and visited[current_y][current_x - 1] == 0:
            unvisited_neighbors.append(0)  # Left
        if current_y > 0 and visited[current_y - 1][current_x] == 0:
            unvisited_neighbors.append(1)  # Top
        if current_x < width - 1 and visited[current_y][current_x + 1] == 0:
            unvisited_neighbors.append(2)  # Right
        if current_y < height - 1 and visited[current_y + 1][current_x] == 0:
            unvisited_neighbors.append(3)  # Bottom

        if unvisited_neighbors:
            direction = random.choice(unvisited_neighbors)
            new_x, new_y = get_opposite_node(direction, (current_x, current_y))

            # Remove walls between current cell and the new cell
            if direction == 0:
                walls[current_y][current_x][0] = 0
                walls[new_y][new_x][2] = 0
            elif direction == 1:
                walls[current_y][current_x][1] = 0
                walls[new_y][new_x][3] = 0
            elif direction == 2:
                walls[current_y][current_x][2] = 0
                walls[new_y][new_x][0] = 0
            elif direction == 3:
                walls[current_y][current_x][3] = 0
                walls[new_y][new_x][1] = 0

            path_stack.append((new_x, new_y))
            visited[new_y][new_x] = 1
        else:
            # Backtrack
            path_stack.pop()
    return walls

#MAZE SOLVING (DIJKSTRA'S)
def dijkstra_search(walls, width, height):
    """
    Finds the shortest path from the top-left (0,0) to the bottom-right
    (width-1, height-1) using a modified Breadth-First Search (BFS)
    which is a special case of Dijkstra's on an unweighted graph.
    Returns a 2D list with the distance from the start node to each cell.
    """
    distances = [[-1 for _ in range(width)] for _ in range(height)]

    # The queue for our search, starting at the top-left corner
    queue = [(0, 0)]
    distances[0][0] = 0

    while queue:
        current_x, current_y = queue.pop(0)
        current_distance = distances[current_y][current_x]

        # Check all four neighbors
        for i in range(4):
            # If there's no wall
            if walls[current_y][current_x][i] == 0:
                next_x, next_y = get_opposite_node(i, (current_x, current_y))

                # If the neighbor hasn't been visited
                if distances[next_y][next_x] == -1:
                    distances[next_y][next_x] = current_distance + 1
                    queue.append((next_x, next_y))

    return distances

def reconstruct_path(distances, width, height, walls):
    """
    Reconstructs the shortest path by backtracking from the end node
    to the start node using the distance grid.
    """
    path = []
    current_x, current_y = width - 1, height - 1

    if distances[current_y][current_x] == -1:
        print("No path found.")
        return path

    path.append((current_x, current_y))

    while (current_x, current_y) != (0, 0):
        current_distance = distances[current_y][current_x]

        # Look for a neighbor with a distance of one less
        for i in range(4):
            if walls[current_y][current_x][i] == 0:
                next_x, next_y = get_opposite_node(i, (current_x, current_y))

                if distances[next_y][next_x] == current_distance - 1:
                    path.append((next_x, next_y))
                    current_x, current_y = next_x, next_y
                    break

    return path[::-1]  # Reverse the path to go from start to end

#MAZE DRAWING
def draw_maze(walls, width, height):
    """Draws the maze using the turtle library."""
    turtle.clear()
    turtle.speed(TURTLE_SPEED)
    turtle.hideturtle()

    # Calculate drawing parameters
    screen_size = 760
    grid_size = screen_size / max(width, height)
    start_x = -screen_size / 2
    start_y = screen_size / 2

    # Draw the outer border
    turtle.penup()
    turtle.goto(start_x, start_y)
    turtle.pendown()
    for _ in range(4):
        turtle.forward(screen_size)
        turtle.right(90)
    turtle.penup()

    # Draw internal walls
    for y in range(height):
        for x in range(width):
            cell_x_pos = start_x + x * grid_size
            cell_y_pos = start_y - y * grid_size

            # Draw right wall
            if walls[y][x][2] == 1:
                turtle.penup()
                turtle.goto(cell_x_pos + grid_size, cell_y_pos)
                turtle.pendown()
                turtle.goto(cell_x_pos + grid_size, cell_y_pos - grid_size)

            # Draw bottom wall
            if walls[y][x][3] == 1:
                turtle.penup()
                turtle.goto(cell_x_pos, cell_y_pos - grid_size)
                turtle.pendown()
                turtle.goto(cell_x_pos + grid_size, cell_y_pos - grid_size)


def draw_path(path, width, height):
    """Draws the solution path on the maze."""
    if not path:
        return

    screen_size = 760
    grid_size = screen_size / max(width, height)
    start_x = -screen_size / 2
    start_y = screen_size / 2

    turtle.penup()
    turtle.color(PATH_COLOR)
    turtle.pensize(grid_size / 4)
    turtle.setheading(0)

    # Go to the start of the path
    start_cell_x, start_cell_y = path[0]
    turtle.goto(start_x + start_cell_x * grid_size + grid_size / 2,
                start_y - start_cell_y * grid_size - grid_size / 2)
    turtle.pendown()

    # Draw the path
    for i in range(1, len(path)):
        next_cell_x, next_cell_y = path[i]
        turtle.goto(start_x + next_cell_x * grid_size + grid_size / 2,
                    start_y - next_cell_y * grid_size - grid_size / 2)
    turtle.penup()

#MAIN PROGRAM
if __name__ == "__main__":
    # Generate the maze
    print("Generating maze...")
    generated_walls = generate_maze(MAZE_WIDTH, MAZE_HEIGHT)

    # Draw the maze
    print("Drawing maze...")
    draw_maze(generated_walls, MAZE_WIDTH, MAZE_HEIGHT)

    # Solve the maze
    print("Solving maze...")
    distances = dijkstra_search(generated_walls, MAZE_WIDTH, MAZE_HEIGHT)

    # Reconstruct and draw the path
    path = reconstruct_path(distances, MAZE_WIDTH, MAZE_HEIGHT, generated_walls)
    draw_path(path, MAZE_WIDTH, MAZE_HEIGHT)

    # Keep the turtle window open until it is closed manually
    turtle.done()