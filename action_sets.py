from OurObsBuilder import OurObsBuilder
import controller_states as cs

ACTION_SET_SIZE = 8

# throttle, steer, yaw, pitch, roll, jump, boost, handbrake
THROTTLE_INDEX = 0
STEER_INDEX = 1
YAW_INDEX = 2
PITCH_INDEX = 3
ROLL_INDEX = 4
JUMP_INDEX = 5
BOOST_INDEX = 6
HANDBRAKE_INDEX = 7

def __get_car_on_ground_action_set_from_controller_state__(controller_state, action_set):
    #copy over values for throttle and steer
    action_set[THROTTLE_INDEX] = controller_state[cs.FOWARDBACKWARD_INDEX]
    action_set[STEER_INDEX] = controller_state[cs.LEFTRIGHT_INDEX]
    #yaw, pitch, and roll cannot be adjusted when car is on ground
    action_set[YAW_INDEX] = 0
    action_set[PITCH_INDEX] = 0
    action_set[ROLL_INDEX] = 0
    #copy over values for jump, boost, and handbrake
    action_set[JUMP_INDEX] = controller_state[cs.JUMP_INDEX]
    action_set[BOOST_INDEX] = controller_state[cs.BOOST_INDEX]
    action_set[HANDBRAKE_INDEX] = controller_state[cs.SHIFT_INDEX]

    return action_set

def __get_car_in_air_action_set_from_controller_state__(controller_state, action_set):
    roll_activated = controller_state[cs.SHIFT_INDEX] == 1
    roll_not_activated = not roll_activated

    #throttle and steer cannot be adjusted when car is in the air, equals 0
    action_set[THROTTLE_INDEX] = 0
    action_set[STEER_INDEX] = 0
    #copy Forward/Back to control pitch
    action_set[PITCH_INDEX] = controller_state[cs.FOWARDBACKWARD_INDEX]

    if roll_activated:
        #yaw cannot be altered when rolling, equals 0
        action_set[YAW_INDEX] = 0
        #copy Left/Right value to control roll
        action_set[ROLL_INDEX] = controller_state[cs.LEFTRIGHT_INDEX]

    elif roll_not_activated:
        #copy Left/Right value to control yaw
        action_set[YAW_INDEX] = controller_state[cs.LEFTRIGHT_INDEX]
        #roll cannot be altered when roll is not activated, equals 0
        action_set[ROLL_INDEX] = 0

    #copy over values for jump and boost
    action_set[JUMP_INDEX] = controller_state[cs.JUMP_INDEX]
    action_set[BOOST_INDEX] = controller_state[cs.BOOST_INDEX]
    #handbrake cannot be used while car is not on ground, equals 0
    action_set[HANDBRAKE_INDEX] = 0

    return action_set


def get_action_set_from_controller_state(controller_state, game_state):
    action_set = [None] * ACTION_SET_SIZE

    car_on_ground = game_state[OurObsBuilder.CAR_ON_GROUND_INDEX]
    car_in_air = not car_on_ground

    if car_on_ground:
        action_set = __get_car_on_ground_action_set_from_controller_state__(controller_state, action_set)
        
        return action_set

    elif car_in_air:
        action_set = __get_car_in_air_action_set_from_controller_state__(controller_state, action_set)
        return action_set