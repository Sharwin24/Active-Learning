# This file is part of the Rectangle Localization Playground project.
# Copyright (C) 2025 Max Muchen Sun

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Author: Max Muchen Sun (msun@u.northwestern.edu)
# Date: 2025-05-25
# Description: Rectangle Localization Playground


import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from IPython.display import display, HTML, clear_output
import io
import base64
import time


class BoxGymBase(gym.Env):
    def __init__(self, seed=None, dt=0.1, sensor_box_size=0.2, num_sensor_samples=10, max_velocity=0.05, inference_num=100):
        super(BoxGymBase, self).__init__()
        self.dt = dt
        self.sensor_box_size = sensor_box_size      
        self.num_sensor_samples = num_sensor_samples
        self.max_velocity = max_velocity            
        self.inference_num = inference_num
        
        self.action_space = gym.spaces.Box(low=-max_velocity, high=max_velocity, shape=(2,), dtype=np.float32)
        
        self.max_dots = 1000
        self.observation_space = gym.spaces.Dict({
            "dots": gym.spaces.Box(low=0.0, high=1.0, shape=(self.max_dots, 3), dtype=np.float32)
        })
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.reset()

        self.pos_dots = np.empty((0,2))
        self.neg_dots = np.empty((0,2))
        self.pred_boxes = None
        
        # Set up an HTML display with a display_id so that we can update it instead of clearing output.
        fig, ax = plt.subplots()
        self.fig = fig 
        self.ax = ax
        self.html_display = display(HTML(""), display_id=True)

    def reset(self):
        # Reset the sensor's position to the center of the unit square.
        self.sensor_pos = self.rng.uniform(low=0.1, high=0.9, size=(2,))
        
        # Create an underlying (hidden) rectangle.
        rng = self.rng
        l_bnd, u_bnd = 0.2, 0.8
        box_center = rng.uniform(low=l_bnd, high=u_bnd, size=(2,))
        edge1_max = min(box_center[0], 1.0-box_center[0]) * 2.0
        edge1_min = 0.1 * edge1_max
        edge2_max = min(box_center[1], 1.0-box_center[1]) * 2.0
        edge2_min = 0.1 * edge2_max
        box_edge1 = rng.uniform(low=edge1_min, high=edge1_max)
        box_edge2 = rng.uniform(low=edge2_min, high=edge2_max)
        box_corner = box_center - np.array([box_edge1, box_edge2]) / 2.0
        self.rect = {"x": box_corner[0], "y": box_corner[1], "w": box_edge1, "h": box_edge2}
        
        # Reset the list of sensor dots
        obs = {
            'sensor_pos': self.sensor_pos.copy(), 
            'curr_positive': np.empty((0,2)), 
            'curr_negative': np.empty((0,2)), 
            'hist_positive': np.empty((0,2)), 
            'hist_negative': np.empty((0,2)), 
            'pred_boxes': np.empty((0,4))
        }  
        self.timestep = 0
        return obs

    def step(self, action):
        # Ensure that the action (a numpy array) does not exceed max_velocity.
        norm = np.linalg.norm(action)
        if norm > self.max_velocity:
            action = action / norm * self.max_velocity
        
        # Update the sensor position using a simple single-integrator dynamic.
        new_pos = np.array(self.sensor_pos) + np.array(action) * self.dt
        # Make sure the new position stays inside [0,1]×[0,1].
        new_pos = np.clip(new_pos, 0.0, 1.0)
        self.sensor_pos = new_pos
        
        # Define the sensor box (centered at sensor_pos).
        half = self.sensor_box_size / 2.0
        left = float(np.clip(self.sensor_pos[0] - half, 0.0, 1.0))
        right = float(np.clip(self.sensor_pos[0] + half, 0.0, 1.0))
        bottom = float(np.clip(self.sensor_pos[1] - half, 0.0, 1.0))
        top = float(np.clip(self.sensor_pos[1] + half, 0.0, 1.0))
        
        # Sample sensor points within the sensor box.
        sensor_dots_w = []
        sensor_dots_b = []
        for _ in range(self.num_sensor_samples):
            sx = self.rng.uniform(left, right)
            sy = self.rng.uniform(bottom, top)
            # Determine the “color” based on whether the point is inside the underlying rectangle.
            if (sx >= self.rect["x"] and sx <= self.rect["x"] + self.rect["w"] and
                sy >= self.rect["y"] and sy <= self.rect["y"] + self.rect["h"]):
                sensor_dots_b.append(np.array([sx, sy]))
            else:
                sensor_dots_w.append(np.array([sx, sy]))
        
        if len(sensor_dots_w) == 0:
            sensor_dots_w = np.array(sensor_dots_w).reshape(0,2)    
        else:
            sensor_dots_w = np.array(sensor_dots_w)
        
        if len(sensor_dots_b) == 0:
            sensor_dots_b = np.array(sensor_dots_b).reshape(0,2)    
        else:
            sensor_dots_b = np.array(sensor_dots_b)
        
        self.timestep += 1
        
        hist_neg_dots = np.concatenate([self.neg_dots, sensor_dots_w], axis=0)
        hist_pos_dots = np.concatenate([self.pos_dots, sensor_dots_b], axis=0)
        pred_boxes = self.inference(hist_neg_dots, hist_pos_dots, self.inference_num)
        
        self.pred_boxes = pred_boxes
        self.neg_dots = hist_neg_dots
        self.pos_dots = hist_pos_dots
        
        cost = np.var(pred_boxes, axis=0).max()
        done = False
        if cost < 1e-05:
            done = True

        obs = {'sensor_pos': new_pos, 'curr_positive': sensor_dots_b, 'curr_negative': sensor_dots_w, 'hist_positive': hist_pos_dots, 'hist_negative': hist_neg_dots, 'pred_boxes': pred_boxes}
        info = {}
        return obs, cost, done, info
    
    def inference(self, neg_dots, pos_dots, N):
        pos_dots = np.asarray(pos_dots)
        neg_dots = np.asarray(neg_dots)
        rng = self.rng
        valid_boxes = []
        candidates_per_iter = int(10 * N)  # generate a batch of candidate boxes at a time
        
        # Case 1: There are black dots.
        if pos_dots.size > 0:
            # Compute minimal bounding box for the black dots.
            xmin = np.min(pos_dots[:, 0])
            ymin = np.min(pos_dots[:, 1])
            xmax = np.max(pos_dots[:, 0])
            ymax = np.max(pos_dots[:, 1])

            # from left
            left_border = 0.0
            neg_dots_from_left = neg_dots[np.where(
                (neg_dots[:,1] > ymin) & (neg_dots[:,1] < ymax) & (neg_dots[:,0] < xmin)
            )[0]]
            if len(neg_dots_from_left) > 0:
                left_border = neg_dots_from_left[:,0].max()
            
            # from right
            right_border = 1.0
            neg_dots_from_right = neg_dots[np.where(
                (neg_dots[:,1] > ymin) & (neg_dots[:,1] < ymax) & (neg_dots[:,0] > xmax)
            )[0]]
            if len(neg_dots_from_right) > 0:
                right_border = neg_dots_from_right[:,0].min()

            # from above
            above_border = 1.0
            neg_dots_from_above = neg_dots[np.where(
                (neg_dots[:,0] > xmin) & (neg_dots[:,0] < xmax) & (neg_dots[:,1] > ymax)
            )[0]]
            if len(neg_dots_from_above) > 0:
                above_border = neg_dots_from_above[:,1].min()

            # from below
            below_border = 0.0
            neg_dots_from_below = neg_dots[np.where(
                (neg_dots[:,0] > xmin) & (neg_dots[:,0] < xmax) & (neg_dots[:,1] < ymin)
            )[0]]
            if len(neg_dots_from_below) > 0:
                below_border = neg_dots_from_below[:,1].max()

            # print(f'borders: {left_border}, {right_border}, {above_border}, {below_border}')
            
            # Allowed ranges: left in [0, xmin], right in [xmax, 1], bottom in [0, ymin], top in [ymax, 1].
            while len(valid_boxes) < N:
                lefts   = rng.uniform(left_border, xmin, candidates_per_iter)
                rights  = rng.uniform(xmax, right_border, candidates_per_iter)
                bottoms = rng.uniform(below_border, ymin, candidates_per_iter)
                tops    = rng.uniform(ymax, above_border, candidates_per_iter)
                boxes = np.stack([lefts, bottoms, rights, tops], axis=1)
                
                if neg_dots.size > 0:
                    valid = np.all((neg_dots[:, 0, None] < boxes[:, 0]) |
                                (neg_dots[:, 0, None] > boxes[:, 2]) |
                                (neg_dots[:, 1, None] < boxes[:, 1]) |
                                (neg_dots[:, 1, None] > boxes[:, 3]),
                                axis=0)
                else:
                    valid = np.ones(candidates_per_iter, dtype=bool)
                
                valid_boxes.extend(boxes[valid].tolist())

        # Case 2: No black dots but there are white dots.
        elif neg_dots.size > 0:
            while len(valid_boxes) < N:
                # Generate candidate boxes anywhere in [0,1]x[0,1]. 
                # For each candidate, pick two x-values and two y-values and sort them.
                xs = np.sort(rng.uniform(0, 1, (candidates_per_iter, 2)), axis=1)
                ys = np.sort(rng.uniform(0, 1, (candidates_per_iter, 2)), axis=1)
                boxes = np.hstack([xs[:, 0:1], ys[:, 0:1], xs[:, 1:2], ys[:, 1:2]])
                
                # Reject any box that has a white dot strictly inside.
                inside = ((neg_dots[:, 0, None] > boxes[:, 0]) &
                        (neg_dots[:, 0, None] < boxes[:, 2]) &
                        (neg_dots[:, 1, None] > boxes[:, 1]) &
                        (neg_dots[:, 1, None] < boxes[:, 3]))
                valid = ~np.any(inside, axis=0)
                valid_boxes.extend(boxes[valid].tolist())
        
        # Case 3: Neither black dots nor white dots.
        else:
            while len(valid_boxes) < N:
                xs = np.sort(rng.uniform(0, 1, (candidates_per_iter, 2)), axis=1)
                ys = np.sort(rng.uniform(0, 1, (candidates_per_iter, 2)), axis=1)
                boxes = np.hstack([xs[:, 0:1], ys[:, 0:1], xs[:, 1:2], ys[:, 1:2]])
                valid_boxes.extend(boxes.tolist())
        
        return np.array(valid_boxes[:N])
    
    def plot(self):
        ax = self.ax 
        
        # Draw the hidden rectangle (grey with opacity)
        rect = plt.Rectangle((self.rect['x'], self.rect['y']), self.rect['w'], self.rect['h'], color='grey', alpha=0.8)
        ax.add_patch(rect)

        # Draw dots
        ax.plot(self.pos_dots[:,0], self.pos_dots[:,1], linestyle='', marker='o', markerfacecolor='k', markeredgecolor='k', markersize=3)
        ax.plot(self.neg_dots[:,0], self.neg_dots[:,1], linestyle='', marker='o', markerfacecolor='w', markeredgecolor='k', markersize=3)

        # Draw MLE bounds
        pred_boxes = np.array(self.pred_boxes)
        vertices1 = pred_boxes[:,[0,1]][:, None, :]
        vertices2 = pred_boxes[:,[0,3]]
        vertices3 = pred_boxes[:,[2,3]]
        vertices4 = pred_boxes[:,[2,1]]
        rects = np.concatenate([vertices1, vertices2[:, None, :], 
                        vertices3[:, None, :], vertices4[:, None, :]], axis=1)
        collection = PolyCollection(rects, facecolors='none', edgecolors='red', alpha=0.1, linewidth=2)
        ax.add_collection(collection)

        # Draw sensor box (black outline)
        half_size = self.sensor_box_size / 2
        sensor_rect = plt.Rectangle((self.sensor_pos[0] - half_size, self.sensor_pos[1] - half_size), self.sensor_box_size, self.sensor_box_size, fill=None, edgecolor='k', linewidth=2)
        ax.add_patch(sensor_rect)
        ax.scatter(self.sensor_pos[0], self.sensor_pos[1], marker='+', c='k', linewidths=2, s=50)

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')

        rect_env = plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill=None, edgecolor='k', alpha=1.0)
        ax.add_patch(rect_env)


    def render(self, mode='notebook'):
        """Update the same display area without clearing the output."""
        self.ax.clear()
        self.plot()

        if mode == 'notebook':
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            self.html_display.update(HTML(f'<img src="data:image/png;base64,{encoded}" />'))
        elif mode == 'gui':
            plt.pause(0.001)
        else:
            print("Invalid rendering mode! Choose \"notebook\" or \"gui\".")