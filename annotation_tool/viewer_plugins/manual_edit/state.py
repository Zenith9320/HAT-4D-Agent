

class ManualEditState:
    def __init__(self):
        self.selected_region = None
        self.selected_indices = []
        self.selection_frame = None
        self.selection_view = None
        self.selection_status_handle = None
        self.key_status_handle = None
        self.action_status_handle = None
        self.edit_history = []
        self.redo_history = []
        self.bbox_frame_handle = None
        self.bbox_box_handle = None
        self.bbox_face_handles = {}
        self.bbox_face_defs = {}
        self.bbox_initial_center = None
        self.bbox_initial_wxyz = None
        self.bbox_dimensions = None
        self.bbox_base_data = None
        self.bbox_base_center = None
        self.bbox_base_wxyz = None
        self.bbox_base_dims = None
        self.bbox_updating = False
        self.transform_mode = "Translate"
        self.axis_mode = "World"
        self.axis_constraint = (False, False, False)
        self.bbox_visible = False