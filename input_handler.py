import ctypes
import ctypes.wintypes as w
import time

user32 = ctypes.windll.user32

VK_W, VK_A, VK_S, VK_D = 0x57, 0x41, 0x53, 0x44
VK_UP, VK_DOWN = 0x26, 0x28
VK_LEFT, VK_RIGHT = 0x25, 0x27
VK_OEM_4, VK_OEM_6 = 0xDB, 0xDD


def held(vk):
    return user32.GetAsyncKeyState(vk) & 0x8000 != 0


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ('wButtons', w.WORD),
        ('bLeftTrigger', ctypes.c_ubyte),
        ('bRightTrigger', ctypes.c_ubyte),
        ('sThumbLX', ctypes.c_short),
        ('sThumbLY', ctypes.c_short),
        ('sThumbRX', ctypes.c_short),
        ('sThumbRY', ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [('dwPacketNumber', w.DWORD), ('Gamepad', XINPUT_GAMEPAD)]


_BTN = {
    'A': 0x1000, 'B': 0x2000,
    'BACK': 0x0020, 'START': 0x0010,
    'DPAD_LEFT': 0x0004, 'DPAD_RIGHT': 0x0008,
    'LB': 0x0100, 'RB': 0x0200,
}


def _load_xinput():
    for name in ('xinput1_4.dll', 'xinput1_3.dll', 'xinput9_1_0.dll'):
        try:
            return ctypes.windll.LoadLibrary(name)
        except OSError:
            pass
    return None


class XINPUT_VIBRATION(ctypes.Structure):
    _fields_ = [('wLeftMotorSpeed', w.WORD), ('wRightMotorSpeed', w.WORD)]


class InputState:
    def __init__(self):
        self.lr = 0.0
        self.fb = 0.0
        self.ud = 0.0
        self.yaw = 0.0
        self.takeoff_land = False
        self.photo = False
        self.record_toggle = False
        self.switch_mode = False
        self.trim_left = False
        self.trim_right = False
        self.trim_reset = False
        self.emergency_land = False
        self.speed_up = False
        self.speed_down = False
        self.quit = False


class InputHandler:
    def __init__(self, deadzone=0.15):
        self.mode = 'keyboard'
        self.deadzone = deadzone
        self._xinput = _load_xinput()
        self._connected = False
        self._prev_w = 0
        self._p = {'tk': False, 'tl': False, 'tr': False}
        self._start_hold = 0.0
        self._vibe_until = 0.0

    def has_gamepad(self):
        if not self._xinput:
            return False
        state = XINPUT_STATE()
        if self._xinput.XInputGetState(0, ctypes.byref(state)) == 0:
            self._connected = True
            return True
        self._connected = False
        return False

    def vibrate(self, left=0, right=0, duration=0.0):
        if not self._xinput:
            return
        v = XINPUT_VIBRATION(left, right)
        self._xinput.XInputSetState(0, ctypes.byref(v))
        if duration:
            self._vibe_until = time.time() + duration

    def switch_mode(self):
        self.mode = 'gamepad' if self.mode == 'keyboard' else 'keyboard'

    def _poll_gamepad(self, st):
        if not self._xinput:
            return
        state = XINPUT_STATE()
        if self._xinput.XInputGetState(0, ctypes.byref(state)) != 0:
            self._connected = False
            return
        self._connected = True
        g = state.Gamepad
        d = self.deadzone

        # normalize from -32768..32767 to -1..1
        # Right stick → moving (lr/fb), Left stick → yaw/ud
        st.lr = (g.sThumbRX / 32768.0) if abs(g.sThumbRX) > 32768 * d else 0.0
        st.fb = -(g.sThumbRY / 32768.0) if abs(g.sThumbRY) > 32768 * d else 0.0
        st.ud = -(g.sThumbLY / 32768.0) if abs(g.sThumbLY) > 32768 * d else 0.0
        st.yaw = -(g.sThumbLX / 32768.0) if abs(g.sThumbLX) > 32768 * d else 0.0

        cur = g.wButtons
        edges = cur & ~self._prev_w

        now = time.time()
        start_pressed = cur & _BTN['START']

        if start_pressed and not (self._prev_w & _BTN['START']):
            self._start_hold = now
        elif start_pressed and (self._prev_w & _BTN['START']):
            if now - self._start_hold >= 0.8:
                st.emergency_land = True
        elif not start_pressed and (self._prev_w & _BTN['START']):
            if now - self._start_hold < 0.8:
                st.takeoff_land = True
            self._start_hold = 0.0

        if edges & _BTN['A']:
            st.photo = True
        if edges & _BTN['B']:
            st.record_toggle = True
        if edges & _BTN['BACK']:
            st.trim_reset = True
        if edges & _BTN['LB']:
            st.speed_down = True
        if edges & _BTN['RB']:
            st.speed_up = True

        if cur & _BTN['DPAD_LEFT']:
            st.trim_left = True
        if cur & _BTN['DPAD_RIGHT']:
            st.trim_right = True

        self._prev_w = cur

        if self._vibe_until and now > self._vibe_until:
            self._xinput.XInputSetState(0, ctypes.byref(XINPUT_VIBRATION(0, 0)))
            self._vibe_until = 0.0

    def poll(self, cv2_key=-1):
        st = InputState()

        if cv2_key == 27:
            st.quit = True
        if cv2_key == ord('r'):
            st.switch_mode = True
        if cv2_key == 9:
            st.trim_reset = True

        sp = cv2_key == 32
        if sp and not self._p['tk']:
            st.takeoff_land = True
        self._p['tk'] = sp

        if self.mode == 'keyboard':
            st.fb = float(held(VK_W)) - float(held(VK_S))
            st.lr = float(held(VK_D)) - float(held(VK_A))
            st.ud = float(held(VK_UP)) - float(held(VK_DOWN))
            st.yaw = float(held(VK_RIGHT)) - float(held(VK_LEFT))

            if cv2_key == ord('q'):
                st.photo = True
            if cv2_key == ord('e'):
                st.record_toggle = True
            if cv2_key == ord('c'):
                st.speed_up = True
            if cv2_key == ord('x'):
                st.speed_down = True

            tl = held(VK_OEM_4)
            tr = held(VK_OEM_6)
            if tl and not self._p.get('tl'):
                st.trim_left = True
            if tr and not self._p.get('tr'):
                st.trim_right = True
            self._p['tl'], self._p['tr'] = tl, tr

        self._poll_gamepad(st)
        return st
