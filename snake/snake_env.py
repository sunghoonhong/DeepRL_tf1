'''
Author: Sunghoon Hong
Title: snake_env.py
Version: 4.0.1
Description: Snake game Environment
Fix:
    (4.0.0)
    PIXEL_SIZE = 20
    Goal Reward = snake.len + 1
    (4.0.1)
    Goal Reward = +1
'''

import os
import time
import random
import pygame as pg
import gym

    
# size
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

GRID_LEN = 20
GRID_SIZE = (GRID_LEN, GRID_LEN)
# time
FPS = 20  # This variable will define how many frames we update per second.
DELAY = 0.1

# color
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

BG_COLOR = BLACK

# action
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
ACTION = ['UP', 'DOWN', 'LEFT', 'RIGHT']

class Head(pg.sprite.Sprite):
    
    def __init__(self,
            init_x=WINDOW_WIDTH//2//GRID_LEN*GRID_LEN,
            init_y=WINDOW_HEIGHT//2//GRID_LEN*GRID_LEN,
            init_direction=None):
        pg.sprite.Sprite.__init__(self)
        self.speed = GRID_LEN
        self.direction = init_direction
        self.rect = pg.Rect((0,0), GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(GREEN)
        self.init_x = init_x
        self.init_y = init_y
        self.rect.topleft = (init_x, init_y)
        self.trace = self.rect

    def reset(self):
        self.speed = GRID_LEN
        self.direction = None
        self.rect.topleft = (self.init_x, self.init_y)
        self.trace = self.rect

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def update(self):
        self.move()

    def move(self):
        self.trace = self.rect
        if self.direction == UP:
            self.rect = self.rect.move(0, -self.speed)
        elif self.direction == DOWN:
            self.rect = self.rect.move(0, self.speed)
        elif self.direction == LEFT:
            self.rect = self.rect.move(-self.speed, 0)
        elif self.direction == RIGHT:
            self.rect = self.rect.move(self.speed, 0)

    def set_direction(self, dir):
        if ((self.direction==RIGHT and dir==LEFT)
            or
            (self.direction==LEFT and dir==RIGHT)
            or
            (self.direction==UP and dir == DOWN)
            or
            (self.direction==DOWN and dir==UP)):
            return
        self.direction = dir
            
    def chain(self, tail):
        self.tail = tail


class Body(pg.sprite.Sprite):

    def __init__(self, head):
        pg.sprite.Sprite.__init__(self)
        self.speed = GRID_LEN
        self.head = head
        self.tail = None
        self.trace = None
        self.rect = pg.Rect(head.rect.topleft, GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(GREEN)

    def move(self):
        self.trace = self.rect
        self.rect = self.head.trace

    def update(self):
        self.move()

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def chain(self, tail):
        self.tail = tail

class Snake():

    def __init__(self, head):
        self.head = head
        self.bodys = pg.sprite.Group()
        self.tail = self.head
        self.life = 1
        self.len = 1

    def reset(self):
        self.head.reset()
        self.bodys.empty()
        self.tail = self.head
        self.life = 1
        self.len = 1

    def push_back(self):
        new_tail = Body(self.tail)
        self.tail.chain(new_tail)
        self.bodys.add(new_tail)
        self.tail = new_tail
        self.len += 1

    def pop_back(self):
        if not self.bodys:
            return
        new_tail = self.tail.head
        self.tail.kill()
        self.tail = new_tail
        self.len -= 1

    def update(self):
        self.head.update()
        self.bodys.update()

    def draw(self, screen):
        self.bodys.draw(screen)
        self.head.draw(screen)
    
    def boundary_collision(self):
        if (self.head.rect.top < 0 or self.head.rect.bottom > WINDOW_HEIGHT
            or self.head.rect.left < 0 or self.head.rect.right > WINDOW_WIDTH):
            return True
        return False
    
    def set_direction(self, key):
        self.head.set_direction(key)


class Goal(pg.sprite.Sprite):

    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.rect = pg.Rect((0, 0), GRID_SIZE)
        self.image = pg.Surface(GRID_SIZE)
        self.image.fill(WHITE)
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)


class Game:

    def __init__(self):
        self.snake = Snake(Head())
        self.goals = pg.sprite.Group()
        self.create_goal()
        self.score = 0
        self.key = None
        self.screen = pg.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.display = False

    def reset(self):
        self.snake.reset()
        self.goals.empty()
        self.create_goal()
        self.key = None
        self.score = 0

    def create_goal(self):
        if not self.goals:
            goal = Goal()
            while True:
                goal.rect.topleft = (
                    random.randrange(0, WINDOW_WIDTH-GRID_LEN, GRID_LEN),
                    random.randrange(0, WINDOW_HEIGHT-GRID_LEN, GRID_LEN)
                )
                if goal.rect.topleft != self.snake.head.rect.topleft:
                    break
            self.goals.add(goal)
        
    def collision(self):
        '''
        return:
            info
        '''
        info = 'ok'
        body_collision = pg.sprite.spritecollide(self.snake.head, self.snake.bodys, False)
        if body_collision:
            self.snake.life -= 1
            info = 'body'
        elif self.snake.boundary_collision():
            self.snake.life -= 1
            info = 'boundary'
        elif pg.sprite.spritecollide(self.snake.head, self.goals, True):
            self.score += 1
            info = 'goal'
            self.snake.push_back()
            
        return info
        
    def update(self):
        '''
        return:
            info
        '''
        if self.key:
            self.snake.set_direction(self.key)
        self.snake.update()
        info = self.collision()
        self.create_goal()
        # done = (self.snake.life <= 0)
        return info

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.goals.draw(self.screen)
        self.snake.draw(self.screen)
    
    def keydown(self, key):
        self.key = key

    def init_render(self):
        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption('Snake v1.0.0')
        self.display = True

    def render(self):
        pg.display.flip()


class Env:
    def __init__(self):
        pg.init()
        self.action_size = 4
        self.game = Game()

    def reset(self):
        self.game.reset()
        self.game.draw()
        observe = pg.surfarray.array3d(self.game.screen)
        return observe, 0, False, None

    def step(self, action):
        if self.game.display:
            pg.event.pump()
        self.game.keydown(action)
        info = self.game.update()
        done = (self.game.snake.life <= 0)
        if info == 'goal':
            reward = 1
        elif info == 'body':
            reward = -1
        elif info == 'boundary':
            reward = -1
        else:
            reward = 0
            
        self.game.draw()
        observe = pg.surfarray.array3d(self.game.screen)
        return observe, reward, done, info

    def init_render(self):
        self.game.init_render()

    def render(self):
        if not self.game.display:
            self.init_render()
        pg.display.flip()

    def snapshot(self):
        pg.image.save(self.game.screen, 'snapshots/'+str(int(time.time()*10000))+'.png')

if __name__ == '__main__':
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (64, 64)
    clock = pg.time.Clock()
    render = True
    game = Game()
    if render:
        game.init_render()
    step = 0
    while game.snake.life > 0:
        time.sleep(DELAY)
        step+=1
        clock.tick(FPS)
        for evt in pg.event.get():
            if evt.type == pg.QUIT:
                quit()
            elif evt.type == pg.KEYDOWN:
                if evt.key == pg.K_UP:
                    game.keydown(UP)
                elif evt.key == pg.K_DOWN:
                    game.keydown(DOWN)
                elif evt.key == pg.K_LEFT:
                    game.keydown(LEFT)
                elif evt.key == pg.K_RIGHT:
                    game.keydown(RIGHT)
                elif evt.key == pg.K_ESCAPE:
                    quit()

        print(game.update())
        game.draw()
        if render:
            game.render()            

    print('Score:', game.score)
