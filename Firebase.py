import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

from datetime import datetime, time



def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.utcnow().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

class DataComposer:
    db = None

    def __init__(self):
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            'projectId': 'phosagro-db-a2e45',
            'storageBucket': 'phosagro-db-a2e45.appspot.com'
        })
        self.db = firestore.client()

    def getCurrentShift(self):
        currentTime = datetime.now()
        shiftType = None

        if is_time_between(time(7, 30), time(19, 30), currentTime.time()):
            currentShift_DT = currentTime.replace(microsecond=0).replace(hour=7).replace(minute=30).replace(
                second=0).isoformat()
            shiftType = 'Day'
        else:
            currentShift_DT = currentTime.replace(microsecond=0).replace(hour=19).replace(minute=30).replace(
                second=0).isoformat()
            shiftType = 'Night'

        return {
            u'dt': currentShift_DT,
            u'type': shiftType
        }

    def CreateCurrentShift(self):
        """
        Call this method to create current shift if it doesn't exist
        """
        shift = self.getCurrentShift()

        shift_DT = shift['dt']
        shift_type = shift['type']

        # запись данных в базу:
        doc_ref = self.db.collection(u'shift').document(shift_DT)
        doc_ref.set({
            u'date': shift_DT,
            u'type': shift_type
        })

    def AddEvent(self, event_Datetime, event_Type, event_wagon, event_trainID, event_State, event_frames):
        """

        @param event_Datetime: DateTime in ISO format (datetime, example: )
        @param event_Type: (string, example: "arrive"/"departure")
        @param event_wagon:(string, example:"12345678")
        @param event_trainID:(int, example:4)
        @param event_State:(string, "emoty"/"full")
        @param event_frames:(dict with two fields: camera name and path to locally saved image, example: {
                u'camera': 'top',
                u'imagePath': 'DatasetUtils/index.jpeg' # path yo local image
            })
        """
        shift = self.getCurrentShift()
        shift_DT = shift['dt']

        eDT = event_Datetime.isoformat() or datetime.utcnow().time().isoformat()
        eType = event_Type or 'n/a'
        eWagon = event_wagon or 'n/a'
        eTrainID = event_trainID or -1
        eState = event_State or 'n/a'

        eFrames = []
        for frame in event_frames:
            cameraName = frame['camera']
            cameraImagePath = frame['imagePath']
            url = self.uploadImage(cameraImagePath, shift_DT+"/"+cameraName+"_"+eDT)
            eFrames.append(url)

        doc_ref = self.db.collection(u'shift').document(shift_DT).collection(u'events').document(eDT)
        doc_ref.set({
            u'frames': eFrames,
            u'state': eState,
            u'type': eType,
            u'wagon': eWagon,
            u'train_id': eTrainID
        })

    def uploadImage(self, image_path, image_name):
        # Put your local file path
        bucket = storage.bucket()
        blob = bucket.blob(image_name+'.png')
        blob.upload_from_filename(image_path)

        # Opt : if you want to make public access from the URL
        blob.make_public()

        return( blob.public_url )


# пример использования:
# DC = DataComposer()
# DC.CreateCurrentShift()
# DC.AddEvent(datetime.now(),
#             "arrive",
#             "12345678",
#             4,
#             "full",
#             [{
#                 u'camera': 'top',
#                 u'imagePath': 'DatasetUtils/index.jpeg' # path yo local image
#             },
#                 {
#                 u'camera': 'mid',
#                 u'imagePath': 'DatasetUtils/index.jpeg' # path to local image
#             }
#             ])
#
# DC.AddEvent()