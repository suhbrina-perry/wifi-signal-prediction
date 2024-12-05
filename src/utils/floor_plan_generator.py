import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from .display_config import DisplayConfig

class FloorPlanGenerator:
    def __init__(self, width=DisplayConfig.INTERNAL_WIDTH, height=DisplayConfig.INTERNAL_HEIGHT):
        self.width = width
        self.height = height
        self.rooms = []
        
    def add_room(self, x, y, width, height, room_type="office"):
        """Add a room to the floor plan."""
        room = {
            'x': x,
            'y': self.height - y - height,  # Flip y-coordinate
            'width': width,
            'height': height,
            'type': room_type
        }
        self.rooms.append(room)
        
    def draw_floor_plan(self, output_path, show_grid=False):
        """Draw and save the floor plan."""
        fig, ax = plt.subplots(figsize=(DisplayConfig.FIGURE_WIDTH, DisplayConfig.FIGURE_HEIGHT))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        # Draw rooms
        for room in self.rooms:
            # Draw room outline
            rect = Rectangle((room['x'], room['y']), 
                           room['width'], room['height'],
                           facecolor='white',
                           edgecolor='black',
                           linewidth=2)
            ax.add_patch(rect)
            
            # Add room label
            ax.text(room['x'] + room['width']/2, 
                   room['y'] + room['height']/2,
                   room['type'],
                   horizontalalignment='center',
                   verticalalignment='center')
        
        # Remove grid if not needed
        if not show_grid:
            ax.grid(False)
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the floor plan
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=DisplayConfig.DPI)
        plt.close()
        
        return output_path

def create_example_floor_plan():
    """Create an example floor plan with typical office layout."""
    generator = FloorPlanGenerator(width=1000, height=800)
    
    # Generate random office layout
    generator.add_room(100, 100, 200, 200, 'office')
    generator.add_room(400, 100, 200, 200, 'meeting')
    generator.add_room(100, 400, 200, 200, 'open_space')
    
    # Save the floor plan
    output_path = generator.draw_floor_plan("example_floor_plan.png")
    return output_path

if __name__ == "__main__":
    # Generate example floor plan
    output_path = create_example_floor_plan()
    print(f"Example floor plan generated: {output_path}")
