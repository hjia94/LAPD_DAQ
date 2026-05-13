"""Reserved EPICS adapter names for the migration path."""


class EpicsScopeAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS scope control is reserved for a future IOC/PV-backed phase.")


class EpicsMotionAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS motion control is reserved for a future IOC/PV-backed phase.")


class EpicsCameraAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS camera control is reserved for a future IOC/PV-backed phase.")


class EpicsTriggerAdapter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("EPICS trigger control is reserved for a future IOC/PV-backed phase.")
