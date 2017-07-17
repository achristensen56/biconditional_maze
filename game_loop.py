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
import sys
import argparse
import DRLAgents as drl 
import pickle
import os
from time import gmtime, strftime

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0, .5)
RED = (255, 0, 0, .5)


DISPLAY = True
VERBOSE = False


textures = {
    'BEDDING': pygame.transform.scale(pygame.image.load('./imgs/bedding.jpg'), (50, 50)),
    'CARDBOARD': pygame.transform.scale(pygame.image.load('./imgs/cardboard.jpg'), (50, 50)),
    'MESH': pygame.transform.scale(pygame.image.load('./imgs/mesh.jpg'), (50, 50)),
    'METAL': pygame.transform.scale(pygame.image.load('./imgs/metal.jpg'), (50, 50))
    }

log_dir = os.path.join('./archive', strftime("%Y%m%d%H%M%S"))

print("the archive directory is: {}".format(log_dir))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def draw_background(screen, game, reward = 0, state = None):
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

                state_num = state_tensor[context, plat, rew_plat_ind[rew], mouse_state]
    '''

    state_tensor = np.arange(0, 104).reshape([2, 2, 2, 13])

    state_ind = np.argmax(state)


    context, plat, rew, mouse_state = np.where(state_tensor == state_ind)

    context = context[0]
    mouse_state = mouse_state[0]
    plat = plat[0]

    targ_plat = game.current_target_platform 
    platform_cond = game.platform_cond

    state_to_pix = {0: [125, 125], 
                    1: [275, 125],
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

    if plat == 0:
        pygame.draw.circle(screen, GREEN, [circ_x, circ_y], 40)
    elif plat == 1:
        pygame.draw.circle(screen, RED, [circ_x, circ_y], 40)


    if rew == 0:
        pygame.draw.circle(screen, BLACK, [200, 100], 20)
    elif rew == 1:
        pygame.draw.circle(screen, BLACK, [200, 400], 20)

    #print(game.last_rewarded_platform)

    pygame.draw.circle(screen, mouse_col, [circ_x, circ_y], 10)

def get_agent(args):

    if args.agent == 'tabular':
        print("succesfully loaded tabular agent")
        return drl.tabular_agent()
    if args.agent == 'basic_Q':
        print("succesfully loaded simple Q agent")
        return drl.DQAgent(state_space = 104, eps = .1)

    if args.agent == 'Q2':
        print("sucesfully loaded 2 layer Q agent")
        return drl.DQAgent2(state_space = 19, eps = .1)


def parse_args():
    parser = argparse.ArgumentParser(description = 'choose the agent type')
    parser.add_argument("agent", help="enter the agent type, options are tabular or basic_Q")
    parser.add_argument("output", help="enter the output type, options are verbose or display")
    args = parser.parse_args()

    if args.output == 'verbose':
        VERBOSE = True
        DISPLAY = False
    else:
        DISPLAY = True

    return args

def main():

    args = parse_args() 


    if DISPLAY:
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

    mouse = get_agent(args)
    state = convert_state_all(game, *game.reset())

    # -------- Main Program Loop -----------

    actions_list = []
    state_list = []
    epoch_rews = 0
    all_rew = []
    i = 0;
    while not done:
        # --- Main event loop
        if DISPLAY:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
         

        action, Q = mouse.select_action(state)

        next_state_, reward, d = game.step(action[0], i)
        
        next_state = convert_state_all(game, *next_state_)

        #the archive function both saves the behavioral data tofile,
        #and also fills the experience replay buffer
        archive(mouse, state, action[0], reward, next_state, d)

        if len(mouse.experience_buffer.buffer) >    5:
            #batch_size = len(mouse.experience_buffer.buffer) // 2 + 1
            batch_size = 3
            mouse.update_network_batch(batch_size, 5)
        else:
            mouse.update_network(reward, state, next_state, action, Q, False)
        

        state = next_state


        if DISPLAY:
            screen.fill(WHITE)

            draw_background(screen, game, reward, convert_state_all(game, *next_state_))
            textsurface = display_stats(screen, game)

        if VERBOSE:
            print("The mouse is in state: {}, and just took action: {}".format(np.argmax(state_con), action))

        state_list.append(np.argmax(state))
        actions_list.append(action[0])

        # --- Go ahead and update the screen with what we've drawn.

        if DISPLAY:
            surf = display_graph(game)
            [screen.blit(line, (400, 100 + j*20)) for j, line in enumerate(textsurface)]
            screen.blit(surf, (400, 180))

        pygame.display.flip()

    
        if (i % 100 == 0):
            all_rew.append(epoch_rews)
            
            if VERBOSE:
                print("last epoch we received {} rewards".format(epoch_rews))
            
            epoch_rews = 0

     
        # --- Limit to 60 frames per second
        #clock.tick(20)
        i+=1

        if i == 200000 and VERBOSE:
            done = True

    plt.figure()
    plt.subplot(131)
    plt.hist(actions_list)
    plt.subplot(132)
    plt.hist(state_list)
    plt.subplot(133)
    plt.plot(all_rew)
    plt.show()
     
    # Close the window and quit.
    pygame.quit()


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

def convert_state_con(game, context, platform_cond, mouse_state):
    
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

    new_state = np.zeros([1, 19]) 

    try:              
        plat = game.platform_cond[vis_plat_ind[mouse_state]]

        new_state[plat] = 1
    except:
        #print("platform exception, mouse state: {}".format(mouse_state))
        pass

    rew = game.last_rewarded_platform
    rew_plat_ind = { 0: 0,
                     1: 0,
                     2: 1,
                     3: 1}
    
    #this is really more like "last side rewarded"
    rew_ = rew_plat_ind[rew]
    #context: {01, 10}, rew_: {01, 10}, plat: {[00, 01, 10]}, mouse_state = identity(12)

    new_state[0, rew_ + 2] = 1 
    new_state[0, context + 4] = 1
    new_state[0, mouse_state + 6] = 1 

    return new_state


def convert_state_all(game, context, platform_cond, mouse_state):
    '''
    (context 2) x (platform condition 2) x (previous reward location 2)
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
    
    state_tensor = np.arange(0, 104).reshape([2, 2, 2, 13])
    
    rew = game.last_rewarded_platform

    rew_plat_ind = { 0: 0,
                     1: 0,
                     2: 1,
                     3: 1}

    #print('context: {}, platform: {}, rewarded_plat: {}, mouse_state: {}'.format(context, plat, rew, mouse_state))

    state_num = state_tensor[context, plat, rew_plat_ind[rew], mouse_state]

    return np.identity(104)[state_num:state_num + 1]

def archive(mouse, state, action, reward, next_state, d, batch_history = [], batch_num = []):
    '''
    in this function I completely abuse my newfound knowledge that 
    python binds default arguments at definition time.
    '''

    if d == 1:
        f = open(os.path.join(log_dir, str(len(batch_num)) + '.pkl'), 'wb')
        pickle.dump(batch_history, f)
        batch_num.append(1)
        mouse.experience_buffer.add(np.array(batch_history))
        batch_history = []
        
    else:
        batch_history.append([np.squeeze(state), np.squeeze(action),
                             reward, np.squeeze(next_state)])





if __name__ == '__main__':
    main()

