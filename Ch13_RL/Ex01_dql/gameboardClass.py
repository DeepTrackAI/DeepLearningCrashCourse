import numpy as np
import random

class TGameBoard:

    def __init__(self,N_row,N_col,tile_size,max_tile_count,agent,stochastic_prob):
        self.N_row=N_row
        self.N_col=N_col
        self.tile_size=tile_size
        self.max_tile_count=max_tile_count
        self.stochastic_prob=stochastic_prob
        self.agent=agent;
        # Create table for game board, entries 1 means occupied, entries -1 means free
        # Use type float32 to simplify conversion to tensors in torch
        self.board=np.empty((N_row,N_col),dtype=np.float32)
        self.cur_tile_type=-1
        self.tile_x=-1
        self.tile_y=-1
        self.tile_orientation=-1
        if self.tile_size==2:
            # Tile set (at most 2 by 2)
            #   x    0 0
            #   x    x x

            #   0 x    x 0
            #   x 0    0 x

            #   x x    x 0    0 x    x x
            #   x 0    x x    x x    0 x

            #   x x
            #   x x
            # Tile structure of tiles[i,j]: The first dimension denotes x value, the length denotes the number of columns taken by the tile. The second dimension consist of pairs giving the y range: first element in the pair is the first row of the tile and the second element second is the last row plus 1 of the tile for the current column
            self.tiles = [
                [[[0,2]], [[0,1],[0,1]]],
                [[[0,1],[1,2]], [[1,2],[0,1]]],
                [[[0,2],[1,2]], [[0,2],[0,1]], [[0,1],[0,2]], [[1,2],[0,2]]],
                [[[0,2],[0,2]]],
            ]
        elif self.tile_size==4:
            # Tile set (at most 4 by 4)
            #   x    
            #   x    
            #   x    
            #   x    x x x x

            #            0 x
            #   x x 0    x x
            #   0 x x    x 0

            #            x 0
            #   0 x x    x x
            #   x x 0    0 x

            #   x x             0 x    
            #   x 0    x 0 0    0 x    x x x
            #   x 0    x x x    x x    0 0 x

            #   x x             x 0    
            #   0 x    x x x    x 0    0 0 x
            #   0 x    x 0 0    x x    x x x

            #            0 x             x 0
            #   0 x 0    x x    x x x    x x
            #   x x x    0 x    0 x 0    x 0

            #   x x
            #   x x
            self.tiles = [
                [[[0,4]],[[0,1],[0,1],[0,1],[0,1]]],
                [[[1,2],[0,2],[0,1]], [[0,2],[1,3]]],
                [[[0,1],[0,2],[1,2]], [[1,3],[0,2]]],
                [[[0,3],[2,3]], [[0,2],[0,1],[0,1]], [[0,1],[0,3]], [[1,2],[1,2],[0,2]]],
                [[[2,3],[0,3]], [[0,2],[1,2],[1,2]], [[0,3],[0,1]], [[0,1],[0,1],[0,2]]],
                [[[0,1],[0,2],[0,1]], [[1,2],[0,3]], [[1,2],[0,2],[1,2]], [[0,3],[1,2]]],
                [[[0,2],[0,2]]],
            ]

        else:
            assert(0)

        # Create predefined tile sequence, used if stochastic_prob=0
        rand_state=random.getstate()
        random.seed(12345)
        self.tile_sequence=[random.randint(0,len(self.tiles)-1) for x in range(self.max_tile_count)]
        random.setstate(rand_state)

        if self.agent is not None:
            self.agent.init_board(self)

        self.fn_restart()

    def fn_restart(self):
        self.gameover=0
        self.tile_count=0
        self.board.fill(-1)
        self.fn_new_tile()

    def fn_new_tile(self):
        if self.tile_count<self.max_tile_count:
            # Choose a random tile with probability stochastic_prob, otherwise take tile from deterministic sequence of tiles
            if random.random()<self.stochastic_prob:
                self.cur_tile_type=random.randint(0,len(self.tiles)-1)
            else:
                self.cur_tile_type=self.tile_sequence[self.tile_count]
            self.tile_count+=1
        else:
            self.gameover=1
        self.tile_x=self.N_col//2
        self.tile_y=self.N_row
        self.tile_orientation=0
        self.agent.get_state()

    def fn_check_boundary(self):
        for xLoop in range(len(self.tiles[self.cur_tile_type][self.tile_orientation])):
            curx=self.tile_x+xLoop
            if(curx<0)or(curx>self.N_col-1):
                return 1
        return 0

    def fn_move(self,new_tile_x,new_tile_orientation):
        if new_tile_orientation>=len(self.tiles[self.cur_tile_type]):
            return 1
        old_tile_x=self.tile_x
        old_tile_orientation=self.tile_orientation
        self.tile_x=new_tile_x
        self.tile_orientation=new_tile_orientation
        if self.fn_check_boundary():
            self.tile_x=old_tile_x
            self.tile_orientation=old_tile_orientation
            return 1
        return 0

    def fn_drop(self):
        curtile=self.tiles[self.cur_tile_type][self.tile_orientation]
        # Find first location where the piece collides with occupied locations on the game board
        self.tile_y=0
        for xLoop in range(len(curtile)):
            curx=(self.tile_x+xLoop)%self.N_col
            # Find first occupied location in this column            
            cury=-1;
            for yLoop in range(self.N_row-1,-1,-1):
                if self.board[yLoop,curx]>0:
                    # Calculate the y position for this column if no other columns are taken into account
                    cury=yLoop+1-curtile[xLoop][0]
                    break
            # Use the largest y position for all columns of the tile
            if self.tile_y<cury:
                self.tile_y=cury

        # Change board entries at the newly placed tile to occupied
        for xLoop in range(len(curtile)):
            if self.tile_y+curtile[xLoop][1]>self.N_row:
                self.gameover=1
                return -100;
            else:
                self.board[self.tile_y+curtile[xLoop][0]:self.tile_y+curtile[xLoop][1],(xLoop+self.tile_x)%self.N_col]=1

        # Remove full lines
        lineCount=0
        for yLoop in range(self.N_row-1,-1,-1):
            if np.sum(np.array(self.board[yLoop,:])>0)==self.N_col:
                lineCount+=1
                for y1Loop in range(yLoop,self.N_row-1):
                    self.board[y1Loop,:]=self.board[y1Loop+1,:]
                self.board[self.N_row-1,:]=-1
        if lineCount>0:
            curReward=10**(lineCount-1)
        else:
            curReward=0
        # Choose the next tile
        self.fn_new_tile()

        return curReward






