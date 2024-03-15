import numpy as np

JENGA_BLOCK_DIM=np.array((.075,.025,.015))

class Tower:
    """
    jenga tower representation
    list of layers, each layer is a list of three booleans
    """
    def __init__(self,blocks=None,spawn_point=None,variance=.002):
        """
        blocks is a list of boolean tuples, representing block positions
        spawn point is the base of tower (xyz)
        variance is the randomness to initalize the tower by
        
        """
        if blocks is None:
            blocks=[[True,True,True] for _ in range(5)]
        self.blocks=blocks
        self.height=len(blocks)
        
        self.block_info=[[None,None,None] for _ in range(self.height)]
        
        if spawn_point is None:
            spawn_point=np.zeros(3)
        
        self.spacing=.005
        
        self.random_init(spawn_point=spawn_point,variance=variance)
    
    def angle_wiggle(self,angle_variance):
        return np.array((0.,0.,np.random.normal(0,angle_variance)))
        
    def pos_wiggle(self,variance):
        return np.random.normal(0,variance,3)
    
    def perfect_init(self):
        self.random_init(variance=0.)
        
    def random_init(self,spawn_point,variance=.002):
        for h in range(self.height):
            for i in range(3):
                if self.blocks[h][i]:
                
                    rot=np.array((0.,0.,(h%2)*np.pi/2)) # rotate if odd level
                    
                    offset=np.zeros(3)
                    offset+=(0,0,h*JENGA_BLOCK_DIM[2]) # height of level
                    width=JENGA_BLOCK_DIM[1]
                    if h%2:
                        offset+=((i-1)*(width+self.spacing),0,0)
                    else:
                        offset+=(0,(i-1)*(width+self.spacing),0)
                    pos=offset+spawn_point
                    
                    pos+=self.pos_wiggle(variance=variance)
                    rot+=self.angle_wiggle(angle_variance=2*np.pi*variance)
                    
                    dic={
                          'pos':pos,
                          'rot':rot,
                        }
                    self.block_info[h][i]=dic
                    
