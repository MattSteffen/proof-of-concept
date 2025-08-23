import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
running = True

color_on = "blue"
color_off = "black"

switches = [color_off, color_off, color_off, color_off, color_off]
button_centers = [(175, 500), (325, 500), (475, 500), (625, 500)]

def toggle_switch(switch_index):
    if switches[switch_index] == color_off:
        switches[switch_index] = color_on
    else:
        switches[switch_index] = color_off


def click_button_1():
    switches_to_turn_on = []
    switches_to_turn_off = []
    for i in range(len(switches)):
        if switches[i] == color_on:
            if i > 0: # TODO: use range(1, len(switches))
                switches_to_turn_off.append(i)
                switches_to_turn_on.append(i-1)
    # turn off first, then turn the others on
    # TODO: check to see if this behavior is desired
    for i in switches_to_turn_off:
        switches[i] = color_off
    for i in switches_to_turn_on:
        switches[i] = color_on

def click_button_2():
    toggle_switch(1)
    toggle_switch(2)
    toggle_switch(3)

def click_button_3():
    toggle_switch(1)
    toggle_switch(3)

def click_button_4():
    for i in range(len(switches)):
        toggle_switch(i)
    # switches_to_turn_on = []
    # switches_to_turn_off = []
    # for i in range(len(switches)):
    #     if switches[i] == color_on:
    #         if i < len(switches) - 1:
    #             switches_to_turn_off.append(i)
    #             switches_to_turn_on.append(i+1)
    # # TODO: check to see if this behavior is desired
    # for i in switches_to_turn_off:
    #     switches[i] = color_off
    # for i in switches_to_turn_on:
    #     switches[i] = color_on
    

# circle_center = (75, 75), mouse_x = 100, mouse_y = 100
def check_click(circle_center, mouse_x, mouse_y, radius=50):
    """Check if mouse click is within circle bounds"""
    distance_squared = (mouse_x - circle_center[0])**2 + (mouse_y - circle_center[1])**2
    return distance_squared <= radius**2



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if check_click(button_centers[0], mouse_x, mouse_y):
                click_button_1()
            if check_click(button_centers[1], mouse_x, mouse_y):
                click_button_2()
            if check_click(button_centers[2], mouse_x, mouse_y):
                click_button_3()
            if check_click(button_centers[3], mouse_x, mouse_y):
                click_button_4()
    
    screen.fill("green")

    # draw switch backgrounds
    pygame.draw.rect(screen, "red", pygame.Rect(50, 50, 100, 200))
    pygame.draw.rect(screen, "red", pygame.Rect(200, 50, 100, 200))
    pygame.draw.rect(screen, "red", pygame.Rect(350, 50, 100, 200))
    pygame.draw.rect(screen, "red", pygame.Rect(500, 50, 100, 200))
    pygame.draw.rect(screen, "red", pygame.Rect(650, 50, 100, 200))

    # draw switches
    pygame.draw.rect(screen, switches[0], pygame.Rect(75, 75, 50, 150))
    pygame.draw.rect(screen, switches[1], pygame.Rect(75+150*1, 75, 50, 150))
    pygame.draw.rect(screen, switches[2], pygame.Rect(75+150*2, 75, 50, 150))
    pygame.draw.rect(screen, switches[3], pygame.Rect(75+150*3, 75, 50, 150))
    pygame.draw.rect(screen, switches[4], pygame.Rect(75+150*4, 75, 50, 150))

    # draw buttons
    pygame.draw.circle(screen, "blue", button_centers[0], 50)
    pygame.draw.circle(screen, "blue", button_centers[1], 50)
    pygame.draw.circle(screen, "blue", button_centers[2], 50)
    pygame.draw.circle(screen, "blue", button_centers[3], 50)


    pygame.display.flip()
    clock.tick(60)

    if switches == [color_on, color_on, color_on, color_on, color_on]:
        print("You win!")
        running = False


pygame.quit()