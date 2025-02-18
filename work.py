import requests

def decode_message(url):
    # Fetch the document content from the URL
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the document.")
        return

    # Parse the content of the document
    content = response.text.strip().split('\n')

    # Initialize an empty dictionary to hold characters and their coordinates
    grid_data = {}
    max_x, max_y = 0, 0

    # Extract characters and their coordinates
    for line in content:
        parts = line.split()
        if len(parts) == 3:  # Expecting three parts: x-coordinate, character, y-coordinate
            x = int(parts[0])
            char = parts[1]
            y = int(parts[2])
            grid_data[(x, y)] = char
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # Create a grid initialized with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Fill the grid with the characters from the document
    for (x, y), char in grid_data.items():
        grid[y][x] = char

    # Print the grid row by row
    for row in grid:
        print(''.join(row))

# Example usage:
url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
decode_message(url)
