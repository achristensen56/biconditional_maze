"""
 Pygame base template for opening a window
 
 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/
 
 Explanation video: http://youtu.be/vRB_983kUMc
"""
 
import pygame
import Maze_Env as ME
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

import DRLAgents as drl 

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

textures = {
    'BEDDING': pygame.transform.scale(pygame.image.load('./imgs/bedding.jpg'), (50, 50)),
    'CARDBOARD': pygame.transform.scale(pygame.image.load('./imgs/cardboard.jpg'), (50, 50)),
    'MESH': pygame.transform.scale(pygame.image.load('./imgs/mesh.jpg'), (50, 50)),
    'METAL': pygame.transform.scale(pygame.image.load('./imgs/metal.jpg'), (50, 50))
    }



def draw_background(screen, game, context= None, platform_cond = None, mouse_state = 0, targ_plat = None, reward = 0):
    '''
            -----             -----
           |  0  |           |  1  |
            -----------------------
           |  4  |  5  |  6  |  7  |
            -----------------------
                            |  12  |
            -----------------------
           |  8  | 9 |  10  |  11  |
            -----------------------
           |  2  |           |  3  |
            -----             -----
    '''

    platform_cond = game.platform_cond

    state_to_pix = {0: [125, 125], 
                    1: [275, 175],
                    4: [125, 175], 
                    5: [175, 175], 
                    6: [225, 175], 
                    7: [275, 175], 
                    12: [275, 225], 
                    11: [275, 275],
                    3: [275, 325],
                    10: [225, 275], 
                    9: [175, 275], 
                    8: [125, 275], 
                    2: [125, 325]}




    circ_x = state_to_pix[mouse_state][0]
    circ_y = state_to_pix[mouse_state][1]

    mouse_col = BLACK

    if reward == 1:
        mouse_col = (0, 100, 0)


    screen.fill(WHITE)


    if context == 0:
        screen.blit(textures['BEDDING'],  (100, 150))
        screen.blit(textures['BEDDING'],  (150, 150))
        screen.blit(textures['BEDDING'],  (200, 150))
        screen.blit(textures['BEDDING'],  (250, 150))
        screen.blit(textures['BEDDING'],  (250, 200))
        screen.blit(textures['BEDDING'],  (250, 250))
        screen.blit(textures['BEDDING'],  (200, 250))
        screen.blit(textures['BEDDING'],  (150, 250))
        screen.blit(textures['BEDDING'],  (100, 250))

    else:
        screen.blit(textures['CARDBOARD'],  (100, 150))
        screen.blit(textures['CARDBOARD'],  (150, 150))
        screen.blit(textures['CARDBOARD'],  (200, 150))
        screen.blit(textures['CARDBOARD'],  (250, 150))
        screen.blit(textures['CARDBOARD'],  (250, 200))
        screen.blit(textures['CARDBOARD'],  (250, 250))
        screen.blit(textures['CARDBOARD'],  (200, 250))
        screen.blit(textures['CARDBOARD'],  (150, 250))
        screen.blit(textures['CARDBOARD'],  (100, 250))


    if targ_plat == 0:
        pygame.draw.rect(screen, (0, 100, 0, .1), (95, 95, 60, 60))
    elif targ_plat == 1:
        pygame.draw.rect(screen, (0, 100, 0, .1), (245, 95, 60, 60))
    elif targ_plat == 2:
        pygame.draw.rect(screen, (0, 100, 0, .1), (95, 295, 60, 60))
    elif targ_plat == 3:
        pygame.draw.rect(screen, (0, 100, 0, .1), (245, 295, 60, 60))

    #draw the goals
    if platform_cond[0] == 0:
        screen.blit(textures['MESH'],  (100, 100))

    elif platform_cond[0] == 1:
        screen.blit(textures['METAL'],  (100, 100))

    if platform_cond[1] == 0:
        screen.blit(textures['MESH'],  (250, 100))
    elif platform_cond[1] == 1:
        screen.blit(textures['METAL'],  (250, 100))

    if platform_cond[2] == 0:
        screen.blit(textures['MESH'],  (100, 300))
    elif platform_cond[2] == 1:
        screen.blit(textures['METAL'],  (100, 300)) 

    if platform_cond[3] == 0:
        screen.blit(textures['MESH'],  (250, 300))
    elif platform_cond[3] == 1:
        screen.blit(textures['METAL'],  (250, 300))



    pygame.draw.circle(screen, mouse_col, [circ_x, circ_y], 10)

 

def main(): 

    pygame.init()
     
    # Set the width and height of the screen [width, height]
    size = (700, 500)
    screen = pygame.display.set_mode(size)
     
    pygame.display.set_caption("My Game")
    pygame.font.init()
     
    # Loop until the user clicks the close button.
    done = False
     
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    game = ME.Maze()
    mouse = drl.DQAgent(eps = .2, state_space = 208)
    state = game.reset()
    state_con = convert_state_all(game, *state)
     
    # -------- Main Program Loop -----------
    i = 0;
    while not done:
        # --- Main event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
     
        # --- Game logic should go here
     
        # --- Screen-clearing code goes here
     
        # Here, we clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
     
        # If you want a background image, replace this clear with blit'ing the
        # background image.
        screen.fill(WHITE)

        action, Q = mouse.select_action(state_con)

        next_state, reward = game.step(action[0])
        next_state_converted = convert_state_all(game, *next_state)

        mouse.update_network(reward, state_con, next_state_converted, action, Q)
        state = next_state
        state_con = next_state_converted

        draw_background(screen, game, *state, game.current_target_platform, reward)
        textsurface = display_stats(screen, game)
     
        # --- Go ahead and update the screen with what we've drawn.
        [screen.blit(line, (400, 100 + i*20)) for i, line in enumerate(textsurface)]
    
        if i % 50 == 0:
            surf = display_graph(game)
        
        screen.blit(surf, (400, 180))

        pygame.display.flip()

     
        # --- Limit to 60 frames per second
        clock.tick(100)
        i+=1
     
    # Close the window and quit.
    #pygame.quit()


def display_stats(screen, game):

    myfont = pygame.font.SysFont('Times New Roman', 15)

    rew_text = ["total rewards: {}".format(game.total_rewards_received),  
               "batch rewards: {}".format(game.batch_rewards_received), 
               "epoch number: {}".format(game.num_batches),
               "trial number: {}".format(game.total_trial_num)]


    textsurface  = [myfont.render(text, False, (0, 0, 0)) for text in rew_text]
    

    return textsurface

def display_graph(game):
    fig = plt.figure(figsize = [3, 2])
    ax = fig.add_subplot(111)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    canvas = agg.FigureCanvasAgg(fig)

    ax.plot(game.reward_history)

    canvas.draw()

    renderer = canvas.get_renderer()

    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    plt.close(fig)

    return pygame.image.fromstring(raw_data, size, "RGB")

def convert_state(context= None, platform_cond = None, mouse_state = 0):
    

    state = np.zeros([1, 19])

    if context == 0:
        state[0, 0] = 1
    if context == 1:
        state[0, 1] = 1

    state[0, 2:6] = platform_cond
    state[0, mouse_state + 6] = 1 

    return state


def convert_state_all(game, context, platform_cond, mouse_state):
    '''
    (context 2) x (platform condition 2) x (previous reward location 4)
    (platform location 13)

    '''

    vis_plat_ind = {0: 0, 
                    1: 1, 
                    2: 2, 
                    3: 3, 
                    4: 0, 
                    5: 0, 
                    6: 1, 
                    7: 1, 
                    8: 2,
                    9: 2, 
                    10: 3, 
                    11: 3}

    try:              
        plat = game.platform_cond[vis_plat_ind[mouse_state]]
    except:
        #print("platform exception, mouse state: {}".format(mouse_state))
        plat = 0
    
    state_tensor = np.arange(0, 208).reshape([2, 2, 4, 13])
    
    rew = game.last_rewarded_platform

    state_num = state_tensor[context, plat, rew, mouse_state]

    return np.identity(208)[state_num:state_num + 1]

if __name__ == '__main__':
    main()

