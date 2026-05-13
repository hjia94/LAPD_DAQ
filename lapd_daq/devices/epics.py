"""Reserved EPICS adapter names for the migration path."""


class EPICSScopeAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS scope control is reserved for a future IOC/PV-backed phase.")


class EPICSMotionAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS motion control is reserved for a future IOC/PV-backed phase.")


class EPICSCameraAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS camera control is reserved for a future IOC/PV-backed phase.")


class EPICSTriggerAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS trigger control is reserved for a future IOC/PV-backed phase.")


EpicsScopeAdapter = EPICSScopeAdapter
EpicsMotionAdapter = EPICSMotionAdapter
EpicsCameraAdapter = EPICSCameraAdapter
EpicsTriggerAdapter = EPICSTriggerAdapter
