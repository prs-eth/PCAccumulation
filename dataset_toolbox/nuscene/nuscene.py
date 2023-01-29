from pathlib import Path
import pickle

class Dataset(object):
    """An abstract class representing a pytorch-like Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    NumPointFeatures = -1
    def __getitem__(self, index):
        """This function is used for preprocess.
        you need to create a input dict in this function for network inference.
        format: {
            anchors
            voxels
            num_points
            coordinates
            if training:
                labels
                reg_targets
            [optional]anchors_mask, slow in SECOND v1.5, don't use this.
            [optional]metadata, in kitti, image index is saved in metadata
        }
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_sensor_data(self, query):
        """Dataset must provide a unified function to get data.
        Args:
            query: int or dict. this param must support int for training.
                if dict, should have this format (no example yet): 
                {
                    sensor_name: {
                        sensor_meta
                    }
                }
                if int, will return all sensor data. 
                (TODO: how to deal with unsynchronized data?)
        Returns:
            sensor_data: dict. 
            if query is int (return all), return a dict with all sensors: 
            {
                sensor_name: sensor_data
                ...
                metadata: ... (for kitti, contains image_idx)
            }
            
            if sensor is lidar (all lidar point cloud must be concatenated to one array): 
            e.g. If your dataset have two lidar sensor, you need to return a single dict:
            {
                "lidar": {
                    "points": ...
                    ...
                }
            }
            sensor_data: {
                points: [N, 3+]
                [optional]annotations: {
                    "boxes": [N, 7] locs, dims, yaw, in lidar coord system. must tested
                        in provided visualization tools such as second.utils.simplevis
                        or web tool.
                    "names": array of string.
                }
            }
            if sensor is camera (not used yet):
            sensor_data: {
                data: image string (array is too large)
                [optional]annotations: {
                    "boxes": [N, 4] 2d bbox
                    "names": array of string.
                }
            }
            metadata: {
                # dataset-specific information.
                # for kitti, must have image_idx for label file generation.
                image_idx: ...
            }
            [optional]calib # only used for kitti
        """
        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

class NuScenesDataset(Dataset):
    NumPointFeatures = 4  # xyz, timestamp. set 4 to use kitti pretrain
    # NameMapping = {
    #     'movable_object.barrier': 'barrier',
    #     'vehicle.bicycle': 'bicycle',
    #     'vehicle.bus.bendy': 'bus',
    #     'vehicle.bus.rigid': 'bus',
    #     'vehicle.car': 'car',
    #     'vehicle.construction': 'construction_vehicle',
    #     'vehicle.motorcycle': 'motorcycle',
    #     'human.pedestrian.adult': 'pedestrian',
    #     'human.pedestrian.child': 'pedestrian',
    #     'human.pedestrian.construction_worker': 'pedestrian',
    #     'human.pedestrian.police_officer': 'pedestrian',
    #     'movable_object.trafficcone': 'traffic_cone',
    #     'vehicle.trailer': 'trailer',
    #     'vehicle.truck': 'truck'
    # }
    NameMapping = {}
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.parked",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        self._root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        self._nusc_infos = data["infos"]
        self._nusc_infos = list(
            sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
        self._metadata = data["metadata"]
        self._class_names = class_names
        self._prep_func = prep_func
        # kitti map: nusc det name -> kitti eval name
        self._kitti_name_mapping = {
            "car": "car",
            "pedestrian": "pedestrian",
        }  # we only eval these classes in kitti
        self.version = self._metadata["version"]
        self.eval_version = "cvpr_2019"
        self._with_velocity = False

    def __len__(self):
        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        from nuscenes.eval.detection.config import eval_detection_configs
        cls_range_map = eval_detection_configs[self.
                                               eval_version]["class_range"]
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = info["gt_names"]
            gt_boxes = info["gt_boxes"]
            num_lidar_pts = info["num_lidar_pts"]
            mask = num_lidar_pts > 0
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            num_lidar_pts = num_lidar_pts[mask]

            mask = np.array([n in self._kitti_name_mapping for n in gt_names],
                            dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            num_lidar_pts = num_lidar_pts[mask]
            gt_names_mapped = [self._kitti_name_mapping[n] for n in gt_names]
            det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            num_lidar_pts = num_lidar_pts[mask]
            # use occluded to control easy/moderate/hard in kitti
            easy_mask = num_lidar_pts > 15
            moderate_mask = num_lidar_pts > 7
            occluded = np.zeros([num_lidar_pts.shape[0]])
            occluded[:] = 2
            occluded[moderate_mask] = 1
            occluded[easy_mask] = 0
            N = len(gt_boxes)
            gt_annos.append({
                "bbox":
                np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                "alpha":
                np.full(N, -10),
                "occluded":
                occluded,
                "truncated":
                np.zeros(N),
                "name":
                gt_names,
                "location":
                gt_boxes[:, :3],
                "dimensions":
                gt_boxes[:, 3:6],
                "rotation_y":
                gt_boxes[:, 6],
            })
        return gt_annos

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        idx = query
        read_test_image = False
        if isinstance(query, dict):
            assert "lidar" in query
            idx = query["lidar"]["idx"]
            read_test_image = "cam" in query

        info = self._nusc_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "token": info["token"]
            },
        }
        lidar_path = Path(info['lidar_path'])
        points = np.fromfile(
            str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = info["timestamp"] / 1e6

        for sweep in info["sweeps"]:
            points_sweep = np.fromfile(
                str(sweep["lidar_path"]), dtype=np.float32,
                count=-1).reshape([-1, 5])
            sweep_ts = sweep["timestamp"] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                "sweep2lidar_rotation"].T
            points_sweep[:, :3] += sweep["sweep2lidar_translation"]
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]

        if read_test_image:
            if Path(info["cam_front_path"]).exists():
                with open(str(info["cam_front_path"]), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": Path(info["cam_front_path"]).suffix[1:],
            }
        res["lidar"]["points"] = points
        if 'gt_boxes' in info:
            mask = info["num_lidar_pts"] > 0
            gt_boxes = info["gt_boxes"][mask]
            if self._with_velocity:
                gt_velocity = info["gt_velocity"][mask]
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': info["gt_names"][mask],
            }
        return res

    def evaluation_kitti(self, detections, output_dir):
        """eval by kitti evaluation tool.
        I use num_lidar_pts to set easy, mod, hard.
        easy: num>15, mod: num>7, hard: num>0.
        """
        print("++++++++NuScenes KITTI unofficial Evaluation:")
        print(
            "++++++++easy: num_lidar_pts>15, mod: num_lidar_pts>7, hard: num_lidar_pts>0"
        )
        print("++++++++The bbox AP is invalid. Don't forget to ignore it.")
        class_names = self._class_names
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None
        gt_annos = deepcopy(gt_annos)
        detections = deepcopy(detections)
        dt_annos = []
        for det in detections:
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.array([0, 0, 50, 50]))
                anno["alpha"].append(-10)
                anno["dimensions"].append(box3d_lidar[j, 3:6])
                anno["location"].append(box3d_lidar[j, :3])
                anno["rotation_y"].append(box3d_lidar[j, 6])
                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                dt_annos.append(anno)
            else:
                dt_annos.append(kitti.empty_result_anno())
            num_example = dt_annos[-1]["name"].shape[0]
            dt_annos[-1]["metadata"] = det["metadata"]

        for anno in gt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self.NameMapping:
                    mapped_names.append(self.NameMapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        for anno in dt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self.NameMapping:
                    mapped_names.append(self.NameMapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        mapped_class_names = []
        for n in self._class_names:
            if n in self.NameMapping:
                mapped_class_names.append(self.NameMapping[n])
            else:
                mapped_class_names.append(n)

        z_axis = 2
        z_center = 0.5
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        result_official_dict = get_official_eval_result(
            gt_annos,
            dt_annos,
            mapped_class_names,
            z_axis=z_axis,
            z_center=z_center)
        result_coco = get_coco_eval_result(
            gt_annos,
            dt_annos,
            mapped_class_names,
            z_axis=z_axis,
            z_center=z_center)
        return {
            "results": {
                "official": result_official_dict["result"],
                "coco": result_coco["result"],
            },
            "detail": {
                "official": result_official_dict["detail"],
                "coco": result_coco["detail"],
            },
        }

    def evaluation_nusc(self, detections, output_dir):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_train",
            "v1.0-trainval": "val",
        }
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None
        nusc_annos = {}
        mapped_class_names = self._class_names
        token2info = {}
        for info in self._nusc_infos:
            token2info[info["token"]] = info
        for det in detections:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                velocity = box.velocity[:2].tolist()
                if len(token2info[det["metadata"]["token"]]["sweeps"]) == 0:
                    velocity = (np.nan, np.nan)
                box.velocity = np.array([*velocity, 0.0])
            boxes = _lidar_nusc_box_to_global(
                token2info[det["metadata"]["token"]], boxes,
                mapped_class_names, "cvpr_2019")
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                velocity = box.velocity[:2].tolist()
                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": velocity,
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": NuScenesDataset.DefaultAttribute[name],
                }
                annos.append(nusc_anno)
            nusc_annos[det["metadata"]["token"]] = annos
        nusc_submissions = {
            "meta": {
                "use_camera": False,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": nusc_annos,
        }
        res_path = Path(output_dir) / "results_nusc.json"
        with open(res_path, "w") as f:
            json.dump(nusc_submissions, f)
        eval_main_file = Path(__file__).resolve().parent / "nusc_eval.py"
        # why add \"{}\"? to support path with spaces.
        cmd = f"python {str(eval_main_file)} --root_path=\"{str(self._root_path)}\""
        cmd += f" --version={self.version} --eval_version={self.eval_version}"
        cmd += f" --res_path=\"{str(res_path)}\" --eval_set={eval_set_map[self.version]}"
        cmd += f" --output_dir=\"{output_dir}\""
        # use subprocess can release all nusc memory after evaluation
        subprocess.check_output(cmd, shell=True)
        with open(Path(output_dir) / "metrics_summary.json", "r") as f:
            metrics = json.load(f)
        detail = {}
        res_path.unlink()  # delete results_nusc.json since it's very large
        result = f"Nusc {version} Evaluation\n"
        for name in mapped_class_names:
            detail[name] = {}
            for k, v in metrics["label_aps"][name].items():
                detail[name][f"dist@{k}"] = v
            tp_errs = []
            tp_names = []
            for k, v in metrics["label_tp_errors"][name].items():
                detail[name][k] = v
                tp_errs.append(f"{v:.4f}")
                tp_names.append(k)
            threshs = ', '.join(list(metrics["label_aps"][name].keys()))
            scores = list(metrics["label_aps"][name].values())
            scores = ', '.join([f"{s * 100:.2f}" for s in scores])
            result += f"{name} Nusc dist AP@{threshs} and TP errors\n"
            result += scores
            result += "\n"
            result += ', '.join(tp_names) + ": " + ', '.join(tp_errs)
            result += "\n"
        return {
            "results": {
                "nusc": result
            },
            "detail": {
                "nusc": detail
            },
        }

    def evaluation(self, detections, output_dir):
        """kitti evaluation is very slow, remove it.
        """
        # res_kitti = self.evaluation_kitti(detections, output_dir)
        res_nusc = self.evaluation_nusc(detections, output_dir)
        res = {
            "results": {
                "nusc": res_nusc["results"]["nusc"],
                # "kitti.official": res_kitti["results"]["official"],
                # "kitti.coco": res_kitti["results"]["coco"],
            },
            "detail": {
                "eval.nusc": res_nusc["detail"]["nusc"],
                # "eval.kitti": {
                #     "official": res_kitti["detail"]["official"],
                #     "coco": res_kitti["detail"]["coco"],
                # },
            },
        }
        return res

class NuScenesDatasetD8(NuScenesDataset):
    """Nuscenes mini train set. only contains ~3500 samples.
    recommend to use this to develop, train full set once before submit.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::8]