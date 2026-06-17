'''
Setup a simple TCP server to send trigger pulses when requested
Created 2024-12 by J. Han
Modified 2025-07-17 by PP

Runs on Raspberry Pi
- Run normal server mode: python pi_server.py
- Run self-test mode: python pi_server.py --test
- Run custom test parameters: python pi_server.py --test --iterations <N> --delay <T>
NOTE:
    GPIO 0 and 1 are reserved for I2C and are not available for general use. 
    GPIO 2 and 3 also have fixed internal pull-up resistors and are not 
       recommended for general use unless the I2C functionality is disabled. 
    GPIO 14 and 15 are typically used for serial communication and are not
       available for general GPIO use unless serial is disabled. 
    GPIO 34-39 are specifically designed as input-only and do not have internal 
       pull-up or pull-down resistors, making them unsuitable for output. 
'''

import socket
import time
import signal
import sys
import argparse
import ctypes
import os.path

IP_ADDR = '192.168.7.38'
IP_ADDR = '127.0.0.1'
PORT    = 54321
TRIG_OUT_GPIO_NUM = 23
TRIG_IN_GPIO_NUM  = 25

class TriggerServer:
    '''
    Trigger server class that manages GPIO pins and network communication for sending trigger pulses.

    This class sets up a TCP server that listens for trigger requests and controls GPIO pins on a Raspberry Pi.
    It uses a C library (gpio_detect.so) for low-level GPIO control.

    Attributes:
        host (str): IP address to bind the server to
        port (int): Port number to listen on
        trig_out_gpio_num (int): GPIO number for trigger output
        trig_in_gpio_num (int): GPIO number for trigger input detection
        running (bool): Server run state
        gpio_lib: Loaded C library for GPIO control
        sock: TCP socket for network communication

    Example:
        server = TriggerServer(host='192.168.1.100', port=5000, 
                             trig_out_gpio_num=25, trig_in_gpio_num=27)
        server.start()  # Starts listening for trigger requests
    '''
    def __init__(self, trig_out_gpio_num, trig_in_gpio_num, host=IP_ADDR, port=PORT):
        self.host = host
        self.port = port
        self.trig_out_gpio_num = trig_out_gpio_num
        self.trig_in_gpio_num  = trig_in_gpio_num
        self.running = True

        gpio_lib_path = "./gpio_detect.so" #  WAS "/home/generalpi/pi_gpio/gpio_detect.so"
        if not os.path.exists(gpio_lib_path):
            raise FileNotFoundError(f'compiled "gpio_detect.c" library not found at {gpio_lib_path}')

        try:
            self.gpio_lib = ctypes.CDLL(gpio_lib_path)
        except OSError as e:
            raise RuntimeError(f'Failed to load compiled "gpio_detect.c" library: {str(e)}')

        self._setup_gpio_functions()

        if self.gpio_lib.initialize_pigpio() < 0:
            raise RuntimeError("Failed to initialize pigpio")

        if self.gpio_lib.setup_gpio_output_pin(self.trig_out_gpio_num) < 0:
            raise RuntimeError(f"Failed to setup output GPIO# {self.trig_out_gpio_num}")
        if self.gpio_lib.setup_gpio_pin(self.trig_in_gpio_num) < 0:
            raise RuntimeError(f"Failed to setup input GPIO# {self.trig_in_gpio_num}")

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as e:
            raise RuntimeError(f"Socket error: {str(e)}")

        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
                
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
        except Exception as e:
            # Clean up if initialization fails
            if hasattr(self, 'gpio_lib'):
                self.gpio_lib.terminate_pigpio()    #probably never gets here
            if hasattr(self, 'sock'):
                self.sock.close()
            raise
    
    def _setup_gpio_functions(self):
        """Setup C function signatures"""
        try:
            # Initialize and cleanup
            self.gpio_lib.initialize_pigpio.restype = ctypes.c_int
            self.gpio_lib.terminate_pigpio.restype = None
            
            # Pin setup
            self.gpio_lib.setup_gpio_pin.argtypes = [ctypes.c_int]
            self.gpio_lib.setup_gpio_pin.restype = ctypes.c_int
            self.gpio_lib.setup_gpio_output_pin.argtypes = [ctypes.c_int]
            self.gpio_lib.setup_gpio_output_pin.restype = ctypes.c_int
            
            # GPIO operations
            self.gpio_lib.wait_for_gpio_high.argtypes = [ctypes.c_int, ctypes.c_int]
            self.gpio_lib.wait_for_gpio_high.restype = ctypes.c_bool
            self.gpio_lib.send_gpio_pulse.argtypes = [ctypes.c_int]
            self.gpio_lib.send_gpio_pulse.restype = None
        except AttributeError as e:
            raise RuntimeError(f"Failed to setup GPIO functions: {str(e)}")
    
    def send_trigger(self):
        """Send a trigger pulse using C function"""
        try:
            self.gpio_lib.send_gpio_pulse(self.trig_out_gpio_num)
            print(F'>>> sent trigger on GPIO# {self.trig_out_gpio_num}', flush=True)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to send trigger: {str(e)}")
    
    def wait_for_trigger(self, timeout=1):
        """Wait for rising edge using C function"""
        try:
            print(f'>>> waiting for trigger on GPIO# {self.trig_in_gpio_num}...', end='', flush=True)
            wait_status = self.gpio_lib.wait_for_gpio_high(self.trig_in_gpio_num, int(timeout * 1000000))  # Convert to microseconds
            if wait_status:
                print('>>> got it', flush=True)
            else:
                print('>>> failed', flush=True)
            return wait_status
        except Exception as e:
            print('failed', flush=True)
            raise RuntimeError(f"Failed to wait for trigger: {str(e)}")
    
    def handle_command(self, command):
        if not command:
            return 'ERR EMPTY_COMMAND'
            
        try:
            print(f'>>> got command {command}', flush=True)
            cmd_parts = command.upper().split()
            cmd = cmd_parts[0]
            
            if cmd == 'TRIG':
                self.send_trigger()
                return 'OK'
            elif cmd == 'STATUS':
                return 'READY'
            elif cmd == 'TEST_INPUT':
                # Parse test parameters
                if len(cmd_parts) < 2:
                    return 'ERR MISSING_GPIO_NUM'
                try:
                    gpio_num = int(cmd_parts[1])
                    iterations = int(cmd_parts[2]) if len(cmd_parts) > 2 else 5
                    delay = float(cmd_parts[3]) if len(cmd_parts) > 3 else 0.1
                except ValueError:
                    return 'ERR INVALID_PARAMETERS'
                    
                success = self.test_gpio_input(gpio_num, iterations, delay)
                return 'TEST_PASS' if success else 'TEST_FAIL'
                
            elif cmd == 'TEST_OUTPUT':
                # Parse test parameters
                if len(cmd_parts) < 2:
                    return 'ERR MISSING_GPIO_NUM'
                try:
                    gpio_num = int(cmd_parts[1])
                    iterations = int(cmd_parts[2]) if len(cmd_parts) > 2 else 5
                    delay = float(cmd_parts[3]) if len(cmd_parts) > 3 else 0.1
                except ValueError:
                    return 'ERR INVALID_PARAMETERS'
                    
                success = self.test_gpio_output(gpio_num, iterations, delay)
                return 'TEST_PASS' if success else 'TEST_FAIL'
                
            elif cmd == 'WAIT_TRIG':
                try:
                    timeout = float(cmd_parts[1]) if len(cmd_parts) > 1 else 1.0
                except ValueError:
                    return 'ERR INVALID_TIMEOUT'
                    
                if self.wait_for_trigger(timeout=timeout):
                    return 'TRIGGERED'
                return 'NO_TRIGGER'
                
            return 'ERR UNKNOWN_CMD'
            
        except Exception as e:
            return f'ERR {str(e)}'
    
    def test_gpio_input(self, gpio_num, iterations=10, delay=0.1):
        """
        Test GPIO input detection on specified gpio_num.
        
        This test verifies the GPIO input detection by:
        1. Waiting for signal on the specified input gpio_num
        2. Reporting detection success/failure
        3. Repeating for specified iterations
        
        Args:
            gpio_num (int): GPIO number to test for input
            iterations (int): Number of detection cycles to run (default: 10)
            delay (float): Delay between attempts in seconds (default: 0.1)
                - Should account for external signal timing
        
        Returns:
            bool: True if all signals were detected, False otherwise
        """
        print(f"\nStarting GPIO input test:")
        print(f"Input gpio_num: {gpio_num}")
        print(f"Iterations: {iterations}")
        print(f"Delay: {delay}s")
        
        try:
            if self.gpio_lib.setup_gpio_pin(gpio_num) < 0:
                raise RuntimeError(f"Failed to setup input gpio_num {gpio_num}")
            
            success_count = 0
            for i in range(iterations):
                print(f"\nTest {i+1}/{iterations}:", end=" ", flush=True)

                try:
                    if self.gpio_lib.wait_for_gpio_high(gpio_num, int(1000000)):  # 1s timeout
                        success_count += 1
                        print("✓ Signal detected", flush=True)
                    else:
                        print("✗ No signal detected", flush=True)
                except Exception as e:
                    print(f"✗ Error during detection: {str(e)}", flush=True)
                
                time.sleep(delay)
            
            print(f"\nTest complete: {success_count}/{iterations} successful", flush=True)
            return success_count == iterations
            
        except Exception as e:
            print(f"\nTest failed: {str(e)}", flush=True)
            return False
    
    def test_gpio_output(self, gpio_num, iterations=10, delay=0.1):
        """
        Test GPIO output triggering on specified gpio_num.
        
        This test verifies the GPIO output by:
        1. Sending trigger pulses on specified output gpio_num
        2. Reporting trigger sent
        3. Repeating for specified iterations
        
        Args:
            gpio_num (int): GPIO number to test for output
            iterations (int): Number of trigger cycles to run (default: 10)
            delay (float): Delay between triggers in seconds (default: 0.1)
                - Must be longer than trigger pulse width (1ms)
        
        Returns:
            bool: True if all triggers were sent successfully, False otherwise
        """
        print(f"\nStarting GPIO output test:")
        print(f"Output gpio_num: {gpio_num}")
        print(f"Iterations: {iterations}")
        print(f"Delay: {delay}s")
        
        try:
            if self.gpio_lib.setup_gpio_output_pin(gpio_num) < 0:
                raise RuntimeError(f"Failed to setup output gpio_num {gpio_num}")

            success_count = 0
            for i in range(iterations):
                print(f"\nTest {i+1}/{iterations}:", end=" ", flush=True)

                try:
                    self.gpio_lib.send_gpio_pulse(gpio_num)
                    success_count += 1
                    print("✓ Trigger sent", flush=True)
                except Exception as e:
                    print(f"✗ Failed to send trigger: {e}", flush=True)
                
                time.sleep(delay)
            
            print(f"\nTest complete: {success_count}/{iterations} successful", flush=True)
            return success_count == iterations
            
        except Exception as e:
            print(f"\nTest failed: {str(e)}", flush=True)
            return False
    
    def start(self):
        try:
            self.sock.listen(1)
            print(f"Server listening on {self.host}:{self.port}", flush=True)
            
            while self.running:
                try:
                    conn, addr = self.sock.accept()
                    print(f"\nsocket connection from {addr}", flush=True)
                    
                    while self.running:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                break
                                
                            command = data.decode('ascii').strip()
                            response = self.handle_command(command)
                            conn.send(f"{response}\n".encode('ascii'))
                            
                        except socket.error as e:
                            print(f"Socket error while handling client: {str(e)}", flush=True)
                            break
                        except Exception as e:
                            print(f"Error handling command: {str(e)}", flush=True)
                            
                except KeyboardInterrupt:
                    print("\nReceived keyboard interrupt...", flush=True)
                    break
                except socket.error as e:
                    print(f"Socket error: {str(e)}", flush=True)
                except Exception as e:
                    print(f"Error: {str(e)}", flush=True)
                finally:
                    if 'conn' in locals():
                        try:
                            conn.close()
                        except:
                            pass
                            
        except Exception as e:
            print(f"Fatal error: {str(e)}", flush=True)
            self.cleanup()
    
    def signal_handler(self, signum, frame):
        print("\nShutting down server...", flush=True)
        self.running = False
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        print("Cleaning up resources...", flush=True)
        try:
            self.gpio_lib.terminate_pigpio()
        except:
            pass
            
        try:
            self.sock.close()
        except:
            pass
            
        print("Server stopped", flush=True)

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
    try:
        server = TriggerServer(trig_out_gpio_num=TRIG_OUT_GPIO_NUM, trig_in_gpio_num=TRIG_IN_GPIO_NUM, 
                             host=IP_ADDR, port=PORT)
    except Exception as e:
        print(f"Failed to initialize server: {str(e)}")
        sys.exit(1)

    print(f"GPIO Trigger Server initialized")
    print(f"Monitoring trigger input on GPIO# {server.trig_in_gpio_num}")
    print(f"Sending trigger output on GPIO# {server.trig_out_gpio_num}")
    print(f"Listening on {server.host}:{server.port}")
    print("Press Ctrl+C to stop.")
        
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        server.cleanup()
