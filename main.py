import pygame
from Lattice import Lattice
from Network import Network
import numpy as np

if __name__ == "__main__":
    pygame.init()

    # Set up the display
    width, height = 500, 500
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Adaptive Network on Hexagonal Lattice")

    L = Lattice(15,15)  
    network = Network(L, sources=[(0, 0)], sinks=[(14, 14)])

    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    break

        if not paused:
            # Fill the screen with a color (RGB)
            screen.fill((0, 0, 0))
            network.update(screen)
            pygame.display.flip()


    pygame.quit()
