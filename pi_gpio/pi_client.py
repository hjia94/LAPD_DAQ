'''
Client interface for communicating with the GPIO trigger server on Raspberry Pi.
Created 2024-12
Modified 2025-07-17
Author: J. Han

This module provides a TriggerClient class for network communication with the Pi server,
plus utility functions for motor control operations. The client can:
- Send trigger commands to the Pi
- Wait for trigger responses from the Pi  
- Run GPIO input/output tests
- Control motor sequences with trigger synchronization

Usage:
    client = TriggerClient()
    client.send_trigger()
    client.wait_for_trigger(timeout=5)

TODO: both class needs proper exit/close
'''

import socket
import time
import select
import pickle
import os
import sys

# Add paths for imports - works regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add parent directory to path if it exists and isn't already in path
if os.path.exists(parent_dir) and parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configuration variables - modify these to match your setup
PI_HOST = '192.168.7.38'      # Pi server IP address
PI_PORT = 54321            # Pi server port (must match pi_server.py)
DROPPER_IP = '192.168.7.99'  
BUFFER_SIZE = 1024         # Socket buffer size
KAPTON_WINDOW_IP = '192.168.7.98'  


class TriggerClient:
    """
    Client for communicating with the GPIO trigger server on Raspberry Pi.
    
    Provides methods to send commands to the Pi server and receive responses.
    Handles connection management, retries, and error handling automatically.
    
    Attributes:
        host (str): Pi server IP address
        port (int): Pi server port number
        BUF_SIZE (int): Socket buffer size for receiving data
    """
    def __init__(self, host=PI_HOST, port=PI_PORT):
        self.host = host
        self.port = port
        self.BUF_SIZE = BUFFER_SIZE
        
    def send_command(self, command, receive=True, retries=30):
        """
        Send command to Pi with error handling and retries.
        
        Creates a new socket connection for each command and closes it afterward.
        This ensures no persistent connections are maintained, improving reliability
        for long-running operations.
        
        Args:
            command (str): Command to send to Pi server
            receive (bool): Whether to wait for and return response
            retries (int): Number of retry attempts on failure
            
        Returns:
            str or None: Server response if receive=True, None otherwise
        """
        for attempt in range(retries):
            s = None
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((self.host, self.port))

                message = f"{command}\n"
                s.send(message.encode('ascii'))

                if receive:
                    readable, _, _ = select.select([s], [], [], 5.0)
                    if readable:
                        data = s.recv(self.BUF_SIZE)
                        return data.decode('ascii').strip()
                    else:
                        raise TimeoutError("No response from server")
                        
                return None
                
            except TimeoutError:
                if attempt == retries - 1:
                    raise TimeoutError(f"Connection timed out after {retries} attempts")
                time.sleep(0.5)
            except ConnectionRefusedError:
                if attempt == retries - 1:
                    raise ConnectionRefusedError(f"Connection refused by {self.host}:{self.port}")
                time.sleep(0.5)
            except Exception as e:
                if attempt == retries - 1:
                    raise Exception(f"Failed to communicate with Pi: {str(e)}")
                time.sleep(0.5)
            finally:
                if s:
                    s.close()
        
        return None
        
    def send_trigger(self):
        """Send trigger command"""
        response = self.send_command('TRIG')
        if response != 'OK':
            raise RuntimeError(f"Unexpected trigger response: {response}")
        return True
    
    def wait_for_trigger(self, timeout=5):
        """
        Wait for trigger detection from Pi server.
        timeout (float): Maximum time to wait in seconds
        Returns True if trigger received, False if timeout
        Raises RuntimeError If server response is unexpected
        """
        response = self.send_command(f'WAIT_TRIG {timeout}')
        if response == 'TRIGGERED':
            return True
        elif response == 'NO_TRIGGER':
            return False
        else:
            raise RuntimeError(f"Unexpected response: {response}")
        
    def get_status(self):
        """
        Get server status and verify connection.
        Returns True if server is ready.
        Raises RuntimeError If server is not ready or unreachable.
        """
        response = self.send_command('STATUS')
        if response != 'READY':
            raise RuntimeError(f"Server not ready: {response}")
        return True

    def trigger_loop(self, operation_func=None, iterations=1000, delay=0.1, timeout=120):
        """Run a loop sending triggers and executing operations
        
        Args:
            operation_func: Function to call between triggers
            iterations: Number of trigger cycles to run
            delay: Delay between triggers in seconds
            timeout: Maximum time to wait for each trigger in seconds
            
        Returns:
            int: Number of completed iterations
        """
        completed = 0
        
        for i in range(iterations):
            try:
                self.send_trigger()

                if not self.wait_for_trigger(timeout=timeout):
                    print(f"\nFailed to receive trigger {i+1}")
                    continue

                if operation_func:
                    operation_func(i)
                    
                completed += 1
                time.sleep(delay)
                
            except Exception as e:
                print(f"\nError in trigger loop: {e}")
                break
                
        return completed

    def test_gpio_input(self, pin, iterations=5, delay=0.1):
        """
        Test GPIO input detection on specified pin
        Args:
            pin (int): GPIO pin number to test for input
            iterations (int): Number of detection cycles to run (default: 5)
            delay (float): Delay between attempts in seconds (default: 0.1)
        Returns:
            bool: True if all signals were detected, False otherwise
        """
        response = self.send_command(f'TEST_INPUT {pin} {iterations} {delay}')
        if response == 'TEST_PASS':
            return True
        elif response == 'TEST_FAIL':
            return False
        else:
            raise RuntimeError(f"Unexpected test response: {response}")

    def test_gpio_output(self, pin, iterations=5, delay=0.1):
        """
        Test GPIO output triggering on specified pin
        Args:
            pin (int): GPIO pin number to test for output
            iterations (int): Number of trigger cycles to run (default: 5)
            delay (float): Delay between triggers in seconds (default: 0.1)
        Returns:
            bool: True if all triggers were sent successfully, False otherwise
        """
        response = self.send_command(f'TEST_OUTPUT {pin} {iterations} {delay}')
        if response == 'TEST_PASS':
            return True
        elif response == 'TEST_FAIL':
            return False
        else:
            raise RuntimeError(f"Unexpected test response: {response}")

    def __enter__(self):
        """Context manager entry - no special processing after __init__()"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - no special processing"""
        pass

    def __del__(self):
        """Destructor - no special processing"""
        pass

class TungstenDropper:
    '''
    This class is used for loading tungsten balls into the dropper using motor.
    Ball count and max ball count are automatically persisted to a pickle cache file.
    '''
    def __init__(self, motor_ip, timeout=15, cache_file=r"C:\Users\daq\Desktop\LAPD_DAQ\tungsten_dropper_state.pkl"):
        from motion import Motor_Control
        self.mc_w = Motor_Control(server_ip_addr=motor_ip, stop_switch_mode=3)
        self.mc_w.motor_speed = 5
        self.timeout = timeout
        self.cache_file = cache_file
        
        # Initialize private variables
        self._ball_count = 0
        self._max_ball_count = 0
        
        # Load state from cache file if it exists
        self._load_state()
        
        self.spt = self.mc_w.steps_per_rev()
        self.one_drop = int(self.spt/12) + 10
        
    def _load_state(self):
        """Load ball count and max ball count from cache file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    state = pickle.load(f)
                    self._ball_count = state.get('ball_count', 0)
                    self._max_ball_count = state.get('max_ball_count', 0)
                    print(f"Loaded state from cache: ball_count={self._ball_count}, max_ball_count={self._max_ball_count}")
            else:
                print("No cache file found, starting with default values")
        except Exception as e:
            print(f"Error loading state from cache: {e}")
            print("Starting with default values")
    
    def _save_state(self):
        """Save ball count and max ball count to cache file."""
        try:
            state = {
                'ball_count': self._ball_count,
                'max_ball_count': self._max_ball_count
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Error saving state to cache: {e}")
    
    @property
    def ball_count(self):
        """Get current ball count."""
        return self._ball_count
    
    @ball_count.setter
    def ball_count(self, value):
        """Set ball count and save to cache."""
        self._ball_count = value
        self._save_state()
    
    @property
    def max_ball_count(self):
        """Get maximum ball count."""
        return self._max_ball_count
    
    @max_ball_count.setter
    def max_ball_count(self, value):
        """Set maximum ball count and save to cache."""
        self._max_ball_count = value
        self._save_state()
        
    def update_ball_count(self):
        """Update ball count based on current motor position."""
        try:
            self.ball_count = int(self.mc_w.current_step() / self.one_drop)

        except Exception as e:
            raise RuntimeError(f"Error updating ball count: {e}")

    def reset_ball_count(self):
        try:
            self.mc_w.set_zero
            self.update_ball_count()
            print(f"Ball count reset to {self.ball_count}")
        except Exception as e:
            print(f"Error resetting ball count: {e}")
            return False
        return True
    
    def set_max_ball_count(self, max_balls):
        self.max_ball_count = max_balls
        print(f"Max ball count set to {self.max_ball_count}")
    
    def load_ball(self):
        try:
            cur_step = self.mc_w.current_step()
            self.mc_w.turn_to(cur_step + self.one_drop)
            time.sleep(0.5) # Wait for motor to rotate
            new_step = self.mc_w.current_step() # check if motor moved
            if new_step == cur_step:
                raise RuntimeError("Motor position did not change after dropper load command. Check what's going on.")
            
            # Update ball count after confirming motor moved
            self.update_ball_count()
            print(f"Ball count: {self.ball_count}/{self.max_ball_count}")
        except KeyboardInterrupt:
            raise KeyboardInterrupt("Tungsten dropper interrupted by user")
        except Exception as e:
            raise RuntimeError(f"Error dropping ball: {e}")

    def rewind_motor(self, steps):
        '''Sometimes when the motor is stuck, we need to rewind it.'''
        try:
            cur_step = self.mc_w.current_step()
            self.mc_w.turn_to(cur_step - steps)
            time.sleep(1) # Wait for motor to rewind
            new_step = self.mc_w.current_step()
            if new_step == cur_step:
                raise RuntimeError("Motor position did not change after rewind command. Check what's going on.")
        except KeyboardInterrupt:
            raise KeyboardInterrupt("Tungsten dropper interrupted by user")
        except Exception as e:
            raise RuntimeError(f"Error rewinding motor: {e}")
        return True
    
    def __enter__(self):
        """Context manager entry - no special processing after __init__()"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - no special processing"""
        pass

    def __del__(self):
        """Destructor - no special processing"""
        pass

def test_dropper(num_drops=10, max_balls=700, timeout=15):
    try:
        print(f"Initializing TungstenDropper...")
        dropper = TungstenDropper(motor_ip=MOTOR_IP, timeout=timeout)
        dropper.set_max_ball_count(max_balls)

        if not dropper.reset_ball_count():
            print(f"ERROR: Failed to reset ball count")
            return False

        print(f"Starting drop sequence: {num_drops} drops")
        
        for i in range(num_drops):
            try:
                dropper.load_ball()
                time.sleep(1)

            except KeyboardInterrupt:
                print("\nTungsten dropper interrupted by user")
                break
            except Exception as e:
                print(f"ERROR: Failed to load ball {i+1}: {e}")
                break
                
        print(f"\nTest completed: {dropper.ball_count} balls dropped")
        return dropper.ball_count
        
    except Exception as e:
        print(f"Failed to initialize TungstenDropper: {e}")
        return 0

def test():
    """
    Test mode: Runs a trigger loop test with the Pi server.
    
    This demonstrates basic client functionality by:
    1. Connecting to the Pi server
    2. Running a loop that sends triggers and waits for responses
    3. Executing a custom operation between each trigger cycle
    
    Useful for testing the GPIO trigger system end-to-end.
    """
    
    def custom_operation(iteration):
        """Stand-in for real work done between triggers."""
        time.sleep(0.05)

    client = TriggerClient(PI_HOST, PI_PORT)

    try:
        print(f"Connecting to GPIO trigger server at {PI_HOST}:{PI_PORT}")
        client.get_status()  # Raises if server not ready
        print("✓ Server connection established")

        print("\nStarting trigger loop test...")
        print("Press Ctrl+C to stop early")
        completed = client.trigger_loop(
            operation_func=custom_operation,
            iterations=1000,
            delay=0.1,
            timeout=120,
        )
        print(f"\nTest completed: {completed} iterations successful")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    
    dropper = TungstenDropper(motor_ip=DROPPER_IP, timeout=15)

    client = TriggerClient(PI_HOST, PI_PORT)

    from motion import Motor_Control
    mc_k = Motor_Control(server_ip_addr=KAPTON_WINDOW_IP, cm_per_turn=0.425,name="kapton window", stop_switch_mode=3)