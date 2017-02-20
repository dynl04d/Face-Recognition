import subprocess
import settings
import time
import datetime as dt
import json
import os
import RPi.GPIO as GPIO  # need to import this lib for RPi GPIO control (only available on RPi)


class UserAuthorizer(object):
    """

    """
    def __init__(self, max_seq=5):
        self._det_users = {}
        self._det_frames = 0
        self._max_det_seq = max_seq

    def update_user_likelihood(self, det_users):
        """

        :param det_users:
        :return:
        """
        access_granted = False
        usr = None
        if self._det_frames == self._max_det_seq:
            access_granted, usr = self._check_access()
            self._det_frames = 0  # reset the authorization
            self._det_users = {}  # reset user list
            print("###--------------------------------###")
        else:
            if len(det_users) > 0:
                self._det_frames += 1
                print("Check Frame: {}".format(self._det_frames))
            for user in det_users:
                if user in self._det_users.keys():
                    self._det_users[user] += 1  # update user count
                else:
                    self._det_users[user] = 1
        return access_granted, usr

    def _check_access(self):
        """
        :return:
        """
        usrs = None
        access = False
        if len(self._det_users.values()) > 0:
            max_val = max(self._det_users.values())
            if max_val > self._max_det_seq * 0.6:
                usrs = [usr for usr, val in self._det_users.iteritems() if val == max_val]
                if usrs[0] != settings.UNKNOWN_USER:
                    self._grant_access(usrs[0])
                    access = True
                else:
                    self._deny_access()
            else:
                self._deny_access()
        return access, usrs

    def _grant_access(self, usr):
        grant_str = "Access granted. Welcome, {0}.".format(usr)
        self._say(grant_str)
        print(grant_str)

    def _deny_access(self):
        deny_str = "Access denied."
        self._say(deny_str)
        print(deny_str)
        return

    def _say(self, text):
        """
        Passes the given text to an external program which converts
        the text to spoken sounds.
        :param text: Text to be spoken.
        :return:
        """
        if settings.USEPI:
            command = subprocess.Popen(['espeak', '-ven', '-a 200', '-p 50', '-s 150', text], stderr=subprocess.PIPE, shell=False)
        else:
            command = subprocess.Popen(['say', text], stderr=subprocess.PIPE, shell=False)
        return


class LockController(object):
    """
    Defines methods which will be used for the
    servo-motor control. NOTE: only used on the RPI

    """
    def __init__(self):
        self._openPos = 7.4
        self._closePos = 12.8
        self._pwmpin = 17
        self._pwm = None

    def setup(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pwmpin, GPIO.OUT)  # set pin to as output
        self._pwm = GPIO.PWM(self._pwmpin, 50)  # create pwm obj with pin at 50Hz
        return

    def clean(self):
        self._pwm.stop()
        self._pwm = None
        GPIO.cleanup()
        return

    def open(self):
        assert self._pwm is None
        self.setup()
        self._pwm.start(self._openPos)
        time.sleep(0.5)
        self.clean()
        return

    def close(self):
        assert self._pwm is None
        self.setup()
        self._pwm.start(self._closePos)
        time.sleep(0.5)
        self.clean()
        return


class AccessLog(object):
    """
    Logs users attempting to access the
    device via facial recognition.
    """
    def __init__(self, access_log):
        self._logfile = access_log

    def log_user(self, user):
        if user is not None:
            with open(self._logfile, 'rb') as data_file:
                access_list = json.load(data_file)
                data_file.close()

            date = dt.datetime.now()
            date_string = str(date.year) + "-" + str(date.month) + "-" + str(date.day)
            time_string = str(date.hour) + ":" + str(date.minute)
            access = {"Accessed User": user[0], "Date": date_string, "Time": time_string}
            access_list.append(access)

            with open(self._logfile, 'w') as jf:
                json.dump(access_list, jf)
                jf.write(os.linesep)
                jf.close()
        return
