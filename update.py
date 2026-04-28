import numpy as np
import ai2thor
import image_resolution

def train_loop(model, agent, scene):
    controller = Controller(width=..., height=..., visibility_distance=...)
    a = model()
    

class BaseAgent:
    def __ini__(self, 
                controller,
                base_downgrade,
                action_space = {
                    "MoveAhead": {}, 
                    "MoveRight": {}, 
                    "MoveLeft": {}, 
                    "MoveBack": {}, 
                    "RotateRight": {"degrees": 45}, 
                    "RotateLeft": {"degrees": 45}, 
                    "LookUp": {"degrees": 15}, 
                    "LookDown": {"degrees": 15},}
                ):
        self.action_space = action_space
        self.controller = controller
        self.lifetime = 0
        self.action_history = []
        
        self.controller(action='LookUp', degrees=0) #dummy action to push event on controller, might not be needed
        self.base_downgrade = base_downgrade
        self.current_downgrade = base_downgrade
        self.current_vision = self.controller.last_event.frame
        
    def update_state(a):
        '''Update controller action based on policy output
        action, and fixed parameters'''
        
        self.controller(action=self.action_list[a], **self.actions_params_list[a])
        self.lifetime += 1
        self.action_history.append(self.action_list[a])
        
    def update_vision():
        self.current_downgrade *= 2 #TODO
    
    def action_select(a):
        
        if a in self.action_space.keys():
            self.current_downgrade = self.base_downgrade
            update_state(a)
        elif a=='Done':
            check_sucess() # TO DEFINE
        elif a=='Look':
            self.current_vision = update_vision()
        
    def current_reward(distance_to_obj, exploration_coeff, sucess, stopped):
        
        reward = -0.001
        
        if done and sucess
        

    