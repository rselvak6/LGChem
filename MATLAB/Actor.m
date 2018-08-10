classdef Actor
    %Actor: 
    
    properties
        inst %return instance of actor
        w_a1 %return first layer of actor weights for NN (size [Nah,m])
        w_a2 %return second layer of actor weights for NN (size [n,Nah])
        q_az %return output value of NN (size [Nah,1])
        mu %return input to action node (size [n,1])
        sigma_az %return input to neurons (size [Nah,1])
        Nah %return number of neurons used in NN
        eta_a %return actor learning rate
    end
    
    methods
        function self = Actor(inst, w_a1, w_a2, eta_a)
            % Constructor- requires instance and weights
            if nargin == 4
                self.inst = inst;
                self.w_a1 = w_a1;
            	self.w_a2 = w_a2;
                self.eta_a = eta_a;
            else
                error('Insufficient arguments for constructor')
            end
        end
        
        function q_az = applywa1(self,x)
            % Apply weights on states x (first NN layer)
            [m,n] = size(x);
            [m1,n1] = size(self.w_a1);
            if nargin > 0
                if n~=1
                    error('States need to entered as a column vector')
                end
                if n1~=m
                    error('Primary weights need to match size of state vector')
                end
                self.Nah = m1;
                self.sigma_az = sum(self.w_a1)*x;
                q_az = tanh(self.sigma_az/2);
            end
        end
        
        function u = applywa2(self)
            % Apply weights on states x (first NN layer)
            if isempty(self.q_az)
                error('q_az has not been calculated')
            end
            [m,n] = size(self.q_az);
            [m1,n1] = size(self.w_a2);
            if nargin > 0
                if n~=1
                    error('q needs to entered as a column vector')
                end
                if n1~=m
                    error('Secondary weights need to match size of q vector')
                end
                self.mu = sum(self.w_a2)*self.q_az;
                u = tanh(self.mu/2);
            end
       end
        
       function Nah = getNNcnt(self)
           % Return count of number of neurons used in NN
           Nah = self.Nah;
       end
       
       function [w_a1,w_a2] = updateWa(self,w_c1,w_c2,x,u)
           % Return actor weight updates
           dJ_du = w_c2*w_c1/2*sech((w_c1*[x;u])/2)^2;
           du_dmu = sech(self.mu/2)^2/2;
           du_dw1 = tanh(self.sigma_az/2);
           w_a1 = self.w_a1-self.eta_a*ones(length(self.w_a1),1);
           w_a2 = self.w_a2-self.eta_a*ones(length(self.w_a2),1);
       end
        
    end
end

