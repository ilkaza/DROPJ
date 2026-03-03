"""
Obstacle Car Racing Environment
-------------------------------

An extended version of the OpenAI Gym CarRacing environment by Oleg Klimov, augmented for safety-oriented
experimentation, with the following obstacles:
- Chuckholes (static obstacles):
- Scripted cars (NPCs) that move slowly and stay on the road.

The agent must drive on the generated track while avoiding these obstacles.
Several modes allow for randomised, discretised, or hand-crafted configurations of the environment.

State Space
    - RGB image of shape STATE_W x STATE_H pixels (default 128x128) - same as Car Racing Environment

Action Space:
    - Continuous Box: [steering, gas, brake] - same as Car Racing Environment

Reward structure
    - +10 for each new tile visited
    - -1 penalty for going off-road (i.e. stepping on grass)

Info Dictionary (returned in each step):
    - 'success': new tile visited
    - 'crash': agent is off-road (on grass)
    - 'chuck': agent stepped on a new chuckhole
    - 'chuck_passed': agent passed a chuckhole
    - 'car': agent collided with a new car
    - 'car_passed': agent passed a car

Termination:
    - Agent goes outside the PLAYFIELD bounds

Key Configuration Flags:
    - ONLY_CHUCKHOLES: Use only chuckholes as obstacles
    - CHUCKHOLES_CARS: Use both chuckholes and cars
"""


from copy import deepcopy as copy
import sys
import math
import numpy as np
import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener
import gym
from gym import spaces
from gym.envs.box2d.obst_car_dynamics import Car
from gym.utils import seeding, EzPickle
import pyglet
from pyglet import gl
import random
import time

# Set one of these
ONLY_CHUCKHOLES = False # True to add only chuckholes (not cars)
CHUCKHOLES_CARS = True # True to add both chuckholes and cars

# Recommended when drawing trajectories from the keyboard to be not too fast
if CHUCKHOLES_CARS:
    target_step_time = 0
elif ONLY_CHUCKHOLES:
    target_step_time = 0
    target_fps = 50
    target_step_time = 1.0 / target_fps

# environment can be randomised completely (colours, car nums and sizes, chuckhole nums and sizes, etc.)
RANDOMISE = False # includes and randomises everything  uniformly
DISCRETISE = False # includes and randomises everything with discrete values
assert ((RANDOMISE or DISCRETISE) and not (ONLY_CHUCKHOLES or CHUCKHOLES_CARS)) or (
        not (RANDOMISE or DISCRETISE) and (ONLY_CHUCKHOLES ^ CHUCKHOLES_CARS))

pyglet.options["debug_gl"] = False

STATE_W = 128
STATE_H = 128
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7 # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH_RANGE = (35, 60)
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

CHUCKHOLE_MODE = "slowdown"      # Options: "block", "slowdown", "visual"
CHUCKHOLES_CENTRE = True # True to place chuckholes in the central lane
CHUCK_DISTANCE = 7 # minimum distance between chuckholes
REVERSE = False # drive right-wise


ROAD_COLOR = [0.4, 0.4, 0.4]
npc_colors = [
    (0.0, 0.0, 0.8),  # Blue
    # (0.0, 0.6, 0.0),  # Green
    # (0.8, 0.8, 0.0),  # Yellow
    # (0.0, 0.8, 0.8),  # Cyan
    # (0.8, 0.0, 0.8),  # Magenta
    # (1.0, 0.5, 0.0),  # Orange
    (0.5, 0.0, 0.8),  # Purple
    # (0.0, 0.5, 0.5),  # Teal
]

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        car1_id = getattr(u1, "car_id", None)
        car2_id = getattr(u2, "car_id", None)

        if car1_id is not None and car2_id is not None and car1_id != car2_id:
            if 0 in (car1_id, car2_id):  # If agent was involved
                other_car_id = car2_id if car1_id == 0 else car1_id
                if other_car_id not in self.env.hitted_cars:
                    self.env.hitted_cars.add(other_car_id)
                    self.env._just_hit_new_car = True  # mark for this step

        # check for chuckhole interaction
        for chuck, obj in [(u1, u2), (u2, u1)]:
            if getattr(chuck, "is_chuckhole", False):
                if CHUCKHOLE_MODE == "slowdown" or CHUCKHOLE_MODE == "visual":
                    car_id = getattr(obj, "wheel_car_id", None) # works for wheel only (not hull)
                    if car_id is not None:
                        if begin:
                            self.env.active_chuckhole_contacts[car_id] += 1
                        else:
                            self.env.active_chuckhole_contacts[car_id] -= 1

                        self.env.slowdown_active[car_id] = self.env.active_chuckhole_contacts[car_id] > 0
                    if car_id == 0:
                        chuck_id = getattr(chuck, "chuck_id", None)
                        if chuck_id is not None and chuck_id not in self.env.visited_chuckholes:
                            self.env.visited_chuckholes.add(chuck_id)
                            self.env._just_visited_new_chuckhole = True
                return  # handled - don’t process further as road tile

        # normal road tile contact handling
        tile = None
        obj = None
        if u1 and "road_friction" in u1.__dict__:
            tile = u1 # new tile
            obj = u2 # wheel
        if u2 and "road_friction" in u2.__dict__:
            tile = u2 # new tile
            obj = u1 # wheel
        if not tile:
            return

        # only real road tiles reach here
        car_id = getattr(obj, "wheel_car_id", None)
        if car_id == 0: # only for wheels
            tile.color[0] = ROAD_COLOR[0] # changes the colour to completely grey
            tile.color[1] = ROAD_COLOR[1]
            tile.color[2] = ROAD_COLOR[2]

        if not obj or "tiles" not in obj.__dict__:
            return

        if car_id == 0:
            if begin: # wheel touched a tile
                self.env.off_road = False # so not on the grass
                obj.tiles.add(tile) # add the touched tile to the specific wheel
                if not tile.road_visited: # if that tile not visited before
                    tile.road_visited = True
                    self.env.reward += 1000.0 / len(self.env.track)
                    self.env.tile_visited_count += 1
            else:
                obj.tiles.discard(tile)
                if len(obj.tiles) == 0:
                    self.env.off_road = True



class ObstCarRacing(gym.Env, EzPickle):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        # self.car = None

        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # steer, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.off_road = None
        self. prev_tile_visited_count = None
        self.succ_rew_bonus = 10
        self.crash_rew_penalty = -1

        self.RECOVER = True  # avoid getting lost completely in grass
        self.CONTROL_SPEED = True  # avoid over-accelerating
        self.INITIAL_ACC = True  # accelerate on first steps
        self.DRAWING_TRAJECTORIES = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        for car in self.cars:
            car.destroy()

        if hasattr(self, "chuckholes"):
            for ch in self.chuckholes:
                self.world.DestroyBody(ch)
            self.chuckholes = []

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - self.TRACK_WIDTH * math.cos(beta1),
                y1 - self.TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + self.TRACK_WIDTH * math.cos(beta1),
                y1 + self.TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - self.TRACK_WIDTH * math.cos(beta2),
                y2 - self.TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + self.TRACK_WIDTH * math.cos(beta2),
                y2 + self.TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)

            t.tile_id = i

            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * self.TRACK_WIDTH * math.cos(beta1),
                    y1 + side * self.TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (self.TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (self.TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * self.TRACK_WIDTH * math.cos(beta2),
                    y2 + side * self.TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (self.TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (self.TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )


        # Create chuckhole obstacles
        excluded_tiles = {track[i] for i in range(-2, 20)} # avoid chuckholes while car starts

        for i in range(self.num_chuckholes):
            max_attempts = 100  # limit of attempts to create a chuckhole
            attempts = 0

            while attempts < max_attempts:
                attempts += 1
                tile = random.choice(track)
                if tile not in excluded_tiles:
                    if RANDOMISE:
                        if DISCRETISE:
                            tile_idx = track.index(tile)
                            # Exclude tile +/- 1 (bounds-safe)
                            for j in range(tile_idx - 1, tile_idx + 2):
                                if 0 <= j < len(track):
                                    excluded_tiles.add(track[j])
                            break
                        else:
                            break
                    elif ONLY_CHUCKHOLES or CHUCKHOLES_CARS:
                        tile_idx = track.index(tile)
                        # Exclude tile +/- CHUCK_DISTANCE
                        for j in range(tile_idx - CHUCK_DISTANCE, tile_idx + CHUCK_DISTANCE + 1):
                            if 0 <= j < len(track):
                                excluded_tiles.add(track[j])
                        break
                    else:
                        tile_idx = track.index(tile)

                        # Exclude tile +/- 2 (bounds-safe)
                        for j in range(tile_idx - 2, tile_idx + 3):
                            if 0 <= j < len(track):
                                excluded_tiles.add(track[j])
                        break

            alpha, beta, x, y = tile
            if RANDOMISE:
                offset = self.np_random.uniform(-self.TRACK_WIDTH / 2, self.TRACK_WIDTH / 2, size=2)
                pos = (x + offset[0], y + offset[1])
            else:
                offset_choices = [
                    (0.0, TRACK_DETAIL_STEP/2),  # centre of the tile
                    (-4.5, TRACK_DETAIL_STEP/2),  # slight left
                    (4.5, TRACK_DETAIL_STEP/2),  # slight right
                ]
                if DISCRETISE:
                    idx = self.np_random.choice([0, 1, 2], p=[0.5, 0.25, 0.25]) # put most in the centre
                elif CHUCKHOLES_CENTRE:
                    idx = 0
                else:
                    idx = self.np_random.choice([0, 1, 2])
                offset = offset_choices[idx]

                dx, dy = offset

                # Rotate offset vector to align with road direction
                offset_world_x = dx * np.cos(beta) - dy * np.sin(beta)
                offset_world_y = dx * np.sin(beta) + dy * np.cos(beta)

                # Apply to global (x, y)
                pos = (x + offset_world_x, y + offset_world_y)


            if RANDOMISE:
                if DISCRETISE:
                    radius = self.np_random.choice([1.1, 1.2, 1.3])
                else:
                    radius = self.np_random.uniform(1.1, 1.3)
            elif ONLY_CHUCKHOLES or CHUCKHOLES_CARS:
                    radius = 1.2
            else:
                radius = 1.2 # can try other values

            shape = Box2D.b2.circleShape(radius=radius)

            if CHUCKHOLE_MODE == "block":
                fixture = fixtureDef(shape=shape)
            elif CHUCKHOLE_MODE in ["slowdown", "visual"]:
                fixture = fixtureDef(shape=shape, isSensor=True)
            else:
                raise ValueError("Invalid CHUCKHOLE_MODE")

            chuck = self.world.CreateStaticBody(position=pos, fixtures=fixture)
            chuck.color = (1.0, 0.6, 0.0)
            chuck.is_chuckhole = True
            chuck.userData = chuck  # so we can detect it in _contact()
            chuck.chuck_id = i
            chuck.tile_index = track.index(tile)
            chuck.radius = radius
            self.chuckholes.append(chuck)

        self.track = track
        return True

    def reset(self):
        self.step_count = 0

        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.prev_tile_visited_count = 0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        self.cars = []
        if RANDOMISE:
            self.num_cars = self.np_random.randint(10, 20)
        elif DISCRETISE:
            self.num_cars = self.np_random.choice([10, 15, 20])
        elif ONLY_CHUCKHOLES:
            self.num_cars = 1 # agent-car
        elif CHUCKHOLES_CARS:
            self.num_cars = 12
        self.chuckholes = []
        if RANDOMISE:
            self.num_chuckholes = self.np_random.randint(15, 35)
        elif DISCRETISE:
            self.num_chuckholes = self.np_random.choice([15, 20, 25, 30, 35])
        elif ONLY_CHUCKHOLES or CHUCKHOLES_CARS:
            self.num_chuckholes = 20

        self.visited_chuckholes = set()
        self.slowdown_active = {i: False for i in range(self.num_cars)}
        self.active_chuckhole_contacts = {i: 0 for i in range(self.num_cars)}

        self.hitted_cars = set()

        self.passed_chuckholes = set()
        self.passed_cars = set()
        self.close_enough_threshold = 20

        self.grass_colors_rgba = [
            [0.4, 0.8, 0.4, 1.0] * 4,  # vibrant green
            # [0.3, 0.6, 0.3, 1.0] * 4,  # dark green
            # [0.7, 0.7, 0.5, 1.0] * 4,  # yellowish dry
            [0.5, 0.4, 0.3, 1.0] * 4,  # dirt
            # [0.9, 0.9, 0.9, 1.0] * 4,  # snow
        ]

        self.bg_patch_colors = [
            [0.4, 0.9, 0.4, 1.0],
            # [0.2, 0.4, 0.2, 1.0],
            # [0.6, 0.6, 0.4, 1.0],
            [0.4, 0.3, 0.2, 1.0],
            # [0.85, 0.85, 0.85, 1.0],
        ]

        if RANDOMISE:
            idx = self.np_random.randint(len(self.grass_colors_rgba))
        elif DISCRETISE or ONLY_CHUCKHOLES or CHUCKHOLES_CARS:
            idx=0
        self.bg_color = self.bg_patch_colors[idx]
        self.grass_color = self.grass_colors_rgba[idx]
        if RANDOMISE:
            if DISCRETISE:
                self.TRACK_WIDTH = self.np_random.choice([35 / SCALE, 40 / SCALE, 45 / SCALE])
            else:
                self.TRACK_WIDTH = self.np_random.uniform(*TRACK_WIDTH_RANGE) / SCALE
        else:
            self.TRACK_WIDTH  = 40 / SCALE

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

        num_track_tiles = len(self.track)
        self.tile_step = num_track_tiles // self.num_cars
        random.shuffle(npc_colors)
        for i in range(self.num_cars):
            tile_index = i * self.tile_step
            angle, x, y = self.track[tile_index][1:4]
            if REVERSE:
                angle += np.pi  # Reverse the car's angle
            if i == 0:
                size = 0.02  # fixed size for agent
            else:
                if RANDOMISE:
                    if DISCRETISE:
                        size = self.np_random.choice([0.017, 0.02, 0.023])
                    else:
                        size = self.np_random.uniform(0.01, 0.03)
                else:
                    size = 0.02
            car = Car(self.world, angle, x, y, i, size)
            if i == 0:
                car.hull.color = (0.8, 0.0, 0.0)  # red - fixed colour for agent
            else:
                if RANDOMISE:
                    car.hull.color = npc_colors[(i - 1) % len(npc_colors)]
                else:
                    car.hull.color = (0.0, 0.0, 0.8)
            self.cars.append(car)

        self.off_road = False

        return self.step(None)[0]

    def follow_tile_policy(self, car, target_speed=15.0):
        """Scripted policy for npc cars to stay on road on a targeted speed"""

        # get car position and angle
        x, y = car.hull.position
        car_angle = car.hull.angle

        # find closest tile on the track
        closest = min(self.track, key=lambda tile: (tile[2] - x) ** 2 + (tile[3] - y) ** 2)
        tile_angle = closest[1]

        # steering: align with tile direction
        if REVERSE:
            angle_diff = (tile_angle - car_angle) % (2 * np.pi) - np.pi
        else:
            angle_diff = (tile_angle - car_angle + np.pi) % (2 * np.pi) - np.pi

        # compute speed
        velocity = car.hull.linearVelocity
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)

        if target_speed == 15.0:
            steer = np.clip(angle_diff * 4.0, -1.0, 1.0)
        else:
            steering_gain = 4.0 / (1.0 + 0.05 * speed)
            steer = np.clip(angle_diff * steering_gain, -1.0, 1.0)

        # smooth gas control to stay near target speed
        gas = np.clip(0.05 * (target_speed - speed), 0.0, 0.1)

        # slow down on curves
        idx = self.track.index(closest)
        if idx + 1 < len(self.track):
            next_angle = self.track[idx + 1][1]
            turn = abs((next_angle - tile_angle + np.pi) % (2 * np.pi) - np.pi)
            gas *= np.clip(1.0 - turn, 0.4, 1.0)

        return steer, gas

    def recovery_policy(self, car, off_road_threshold=8.0):
        """Recovery policy to avoid getting completely lost in the grass"""
        x, y = car.hull.position
        closest = min(self.track, key=lambda tile: (tile[2] - x) ** 2 + (tile[3] - y) ** 2)
        tile_x, tile_y = closest[2], closest[3]
        tile_beta = closest[1]

        distance = np.sqrt((tile_x - x) ** 2 + (tile_y - y) ** 2)
        if distance < off_road_threshold:
            self.phase1 = True
            self.side_decided = False
            self.steers = []
            return None  # close enough, no recovery needed

        if self.phase1:
            velocity = car.hull.linearVelocity
            speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)
            if speed > 30:
                brake = 1.0
            else:
                brake = 0
            alignment_thresh = 0.05  # ~5 degrees

            angle_diff = (tile_beta - car.hull.angle + np.pi) % (2 * np.pi) - np.pi

            steer, gas = self.follow_tile_policy(car, target_speed=50.0)
            self.steers.append(steer)
            if abs(angle_diff) < alignment_thresh:
                self.phase1 = False
        else: # phase2
            if not self.side_decided:
                positives = sum(1 for x in self.steers if x > 0)
                negatives = sum(1 for x in self.steers if x < 0)

                self.side = 1 if positives > negatives else -1

                self.side_decided = True

            steer = np.clip(0.05 * self.side, -0.3, 0.3)
            gas = 0.2
            brake = 0.0

        return steer, gas, brake

    def get_closest_tile_index(self, car):
        x, y = car.hull.position
        return min(
            range(len(self.track)),
            key=lambda idx: (self.track[idx][2] - x) ** 2 + (self.track[idx][3] - y) ** 2)

    def step(self, action):
        self.step_count += 1

        if self.DRAWING_TRAJECTORIES:
            start_time = time.perf_counter()

        self._just_visited_new_chuckhole = False
        self._just_hit_new_car = False

        # update each car's current tile index
        self.car_tile_indices = {i: self.get_closest_tile_index(car) for i, car in enumerate(self.cars)}
        self.agent_tile_index = self.car_tile_indices[0]

        for i, car in enumerate(self.cars):
            if i == 0:
                # Control agent car
                if action is not None:
                    if self.step_count <= 50 and self.INITIAL_ACC:
                        # accelerate in first steps in straight road
                        car.steer(0.0)
                        car.gas(0.3)
                        car.brake(0.0)
                    else:
                        if self.RECOVER or self.CONTROL_SPEED:
                            recovery_cmd = None
                            if self.RECOVER:
                                recovery_cmd = self.recovery_policy(self.cars[0])
                                if recovery_cmd is not None:
                                    steer, gas, brake = recovery_cmd
                                    car.steer(steer)
                                    car.gas(gas)
                                    car.brake(brake)
                                else:
                                    car.gas(action[1])
                                    car.steer(-action[0])
                                    car.brake(action[2])
                            if self.CONTROL_SPEED and recovery_cmd is None:
                                velocity = car.hull.linearVelocity
                                speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)

                                # cap gas based on speed
                                speed_limit = 40.0
                                if speed > speed_limit:
                                    gas = np.clip(0.05 * (speed_limit - speed), 0.0, 0.1)
                                    car.gas(gas)
                                else:
                                    car.gas(action[1])
                                car.steer(-action[0])
                                car.brake(action[2])
                        else:
                            car.gas(action[1])
                            car.steer(-action[0])
                            car.brake(action[2])
            else:
                # npc cars - want them to keep a distance from the agent-car and from each other
                my_idx = self.car_tile_indices[i]
                should_move = False

                # check agent behind
                agent_idx = self.agent_tile_index

                diff_from_agent = (my_idx - agent_idx) % len(self.track)
                if 0 <= diff_from_agent < 13 or diff_from_agent > len(self.track) - 13:
                    should_move = True
                else:
                    # check other NPCs behind
                    for j in range(1, len(self.cars)):
                        if j == i:
                            continue
                        other_idx = self.car_tile_indices[j]
                        if REVERSE:
                            diff_from_other = (other_idx - my_idx) % len(self.track)
                        else:
                            diff_from_other = (my_idx - other_idx) % len(self.track)
                        if 0 < diff_from_other < self.tile_step:
                            should_move = True
                            break

                if should_move:
                    for j in range(1, len(self.cars)):
                        if j == i:
                            continue
                        other_idx = self.car_tile_indices[j]
                        if REVERSE:
                            front_diff = (my_idx - other_idx) % len(self.track)
                        else:
                            front_diff = (other_idx - my_idx) % len(self.track)
                        if 0 < front_diff < 5:
                            should_move = False
                            break

                if should_move:
                    steer, gas = self.follow_tile_policy(car)
                    car.steer(steer)
                    car.gas(gas)
                    car.brake(0.0)
                else:
                    # stay still
                    car.steer(0.0)
                    car.gas(0.0)
                    car.brake(1.0)

            car.step(1.0 / FPS)

            if self.slowdown_active.get(i, False) and CHUCKHOLE_MODE == "slowdown":
                v = car.hull.linearVelocity
                car.hull.linearVelocity = Box2D.b2Vec2(v.x * 0.6, v.y * 0.6)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self.render("state_pixels") # 96, 96, 3

        step_reward = 0
        done = False
        info = {}

        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            # self.car.fuel_spent = 0.0
            self.cars[0].fuel_spent = 0.0
            # step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track): # if visited all tiles, then terminate
                done = True
            x, y = self.cars[0].hull.position
            oob = abs(x) > PLAYFIELD or abs(y) > PLAYFIELD
            done = oob # terminate only when oob
            if oob or self.off_road:
                step_reward = self.crash_rew_penalty
                info['crash'] = True
            elif self.tile_visited_count > self.prev_tile_visited_count:
                step_reward = self.succ_rew_bonus
                info['succ'] = True

        self.prev_tile_visited_count = copy(self.tile_visited_count)

        # check proximity to chuckholes and cars
        for chuck in self.chuckholes:
            if chuck not in self.passed_chuckholes:
                distance_to_chuck = np.sqrt(
                    (self.cars[0].hull.position[0] - chuck.position[0]) ** 2 +
                    (self.cars[0].hull.position[1] - chuck.position[1]) ** 2)
                if distance_to_chuck < self.close_enough_threshold:
                    self.passed_chuckholes.add(chuck)
                    info['chuck_passed'] = True

        for car in self.cars:
            if car != self.cars[0] and car not in self.passed_cars:
                distance_to_car = np.sqrt(
                    (self.cars[0].hull.position[0] - car.hull.position[0]) ** 2 +
                    (self.cars[0].hull.position[1] - car.hull.position[1]) ** 2)
                if distance_to_car < self.close_enough_threshold:
                    self.passed_cars.add(car)
                    info['car_passed'] = True

        if self._just_visited_new_chuckhole:
            info['chuck'] = True

        if self._just_hit_new_car:
            info['car'] = True

        # Recommended when drawing trajectories from the keyboard
        if self.DRAWING_TRAJECTORIES:
            elapsed = time.perf_counter() - start_time
            time_to_sleep = max(0.0, target_step_time - elapsed)
            time.sleep(time_to_sleep)

        return self.state, step_reward, done, info

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = self.cars[0].hull.position[0]
        scroll_y = self.cars[0].hull.position[1]
        angle = -self.cars[0].hull.angle
        vel = self.cars[0].hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        for car in self.cars:
            car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        if mode != "state_pixels":
            self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def draw_circle(self, polygons_, colors, center, inner_radius, outer_radius, num_segments, color):
        """Draw a filled circle and an outer ring by appending quads to polygons_ and colors"""
        for i in range(num_segments):
            theta1 = 2.0 * math.pi * i / num_segments
            theta2 = 2.0 * math.pi * (i + 1) / num_segments

            p1 = (center[0] + outer_radius * math.cos(theta1),
                center[1] + outer_radius * math.sin(theta1),
                0)

            p2 = (center[0] + outer_radius * math.cos(theta2),
                center[1] + outer_radius * math.sin(theta2),
                0)

            p3 = (center[0] + inner_radius * math.cos(theta2),
                center[1] + inner_radius * math.sin(theta2),
                0)

            p4 = (center[0] + inner_radius * math.cos(theta1),
                center[1] + inner_radius * math.sin(theta1),
                0)

            polygons_.extend(p1 + p2 + p3 + p4)
            colors.extend(color * 4)

    def render_road(self):
        colors = copy(self.grass_color)
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        k = PLAYFIELD / 20.0
        colors.extend(self.bg_color * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        if hasattr(self, "chuckholes"):
            for chuck in self.chuckholes:
                if not chuck.fixtures:
                    continue
                radius = chuck.fixtures[0].shape.radius
                center = chuck.position
                num_segments = 15

                # inner orange fill
                fill_color = (1.0, 0.6, 0.0, 1.0)
                self.draw_circle(
                    polygons_, colors,
                    center=center,
                    inner_radius=0.01,
                    outer_radius=radius,
                    num_segments=num_segments,
                    color=fill_color)

                # outer black ring
                stroke_color = (0.0, 0.0, 0.0, 1.0)
                self.draw_circle(
                    polygons_, colors,
                    center=center,
                    inner_radius=radius,
                    outer_radius=radius + 0.3,  # stroke thickness
                    num_segments=num_segments,
                    color=stroke_color)


        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)
        )  # gl.GL_QUADS,
        vl.draw(gl.GL_QUADS)
        vl.delete()


    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.cars[0].hull.linearVelocity[0])
            + np.square(self.cars[0].hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.cars[0].wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.cars[0].wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.cars[0].wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.cars[0].wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.cars[0].wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.cars[0].hull.angularVelocity, (1, 0, 0))

        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)
        )
        vl.draw(gl.GL_QUADS)
        vl.delete()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = ObstCarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
