# -*- coding: utf-8 -*-

from pfe.services.image_grab.definitions import ImageGrabData
from pfe.services.image_grab.grab.grab_result import GrabResult
from pfe.services.image_grab.ice import ice_image_grab


#
# First item ('JobFinished.errors') in case of an image grab.
#
MSG_GRAB_TRIGGER_TYPE = {
    ice_image_grab.TriggerType.UNKNOWN: "Unknown",
    ice_image_grab.TriggerType.MANUAL: "Image grab",
    ice_image_grab.TriggerType.TRIGGERED: "Triggered image grab",
    ice_image_grab.TriggerType.FMALERT: "FM triggered image grab",
    ice_image_grab.TriggerType.FROMFILE: "Image import from file",
    ice_image_grab.TriggerType.MULTIMANUAL: "Multi image grab",
    ice_image_grab.TriggerType.MULTISCHEDULED: "Multi scheduled image grab",
    ice_image_grab.TriggerType.MULTITRIGGERED: "Multi triggered",
    ice_image_grab.TriggerType.AIADDON: "AI addon",
}

#
# First item ('JobFinished.errors') in case of an overlay request.
#
MSG_OVERLAY_TYPE = {
    ImageGrabData.OVERALL_SEGMENTATION_OVERLAY: "Segmentation overlay request",
    ImageGrabData.OVERALL_EJECTS_OVERLAY: "Ejections overlay request",
    ImageGrabData.OVERALL_CONTOURS_OVERLAY: "Contours overlay request",
    ImageGrabData.OVERALL_ALGORITHMS_OVERLAY: "Algorithms overlay request",
    ImageGrabData.BACKGROUND_OVERLAY: "Background overlay request",
    ImageGrabData.SELECTED_REGION_OVERLAY: "Detailed shape info request"
}


#
# Error messages based on GrabResult (e.g. used by 'JobFinished.errors').
#
MSG_GRAB_ERRORS = {
    GrabResult.SUCCESS: "Image grab was successful.",
    GrabResult.TIMEOUT: "Image grab timed out. "
                        "Timeouts for image grabbing are set to {} seconds for non-triggered image grabs "
                        "and {} seconds for triggered image grabs.".format(
                            ImageGrabData.GRAB_TIMEOUT,
                            ImageGrabData.TRIGGERED_GRAB_TIMEOUT),
    GrabResult.CANCELLED: "Image grab was cancelled.",
    GrabResult.NOT_A_FILE: "Image grab failed due to an I/O error. Logs should be consulted.",
    GrabResult.FAILED: "Image grab failed due to an unknown error. Logs should be consulted.",
    GrabResult.CIP_RUNNING: "Image grab was cancelled because a CIP cycle was active. Please try again.",
}

#
# Overlay error messages (e.g. used by 'JobFinished.errors').
#
SIM_CLTB_RAW_IMAGE_ERROR = "Failed to create raw image. Logs should be consulted."

SIM_SLTB_NO_FILE_ERROR = "{name} overlay for attribute {attribute} failed: no file was generated."
SIM_SLTB_NO_PROJECT_ERROR = "{name} overlay for attribute {attribute} failed: no project."
SIM_SLTB_UNKNOWN_ERROR = "{name} overlay for attribute {attribute} failed: exception occurred."
