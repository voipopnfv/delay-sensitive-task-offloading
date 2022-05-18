from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import time


############ by Ren-Jie ##################
def output_xml_and_img(boxes, scores, classes, image_np, category_index, detected_output_dir): 
    count = 0
    for score in scores:
        if score > 0.5:
            count = count + 1
            
            localtime = time.localtime(time.time())
            localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    
            annotation = ET.Element("annotation")
            filename = ET.SubElement(annotation, "filename")
            filename.text = str(localtime)
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(image_np.shape[1])
            ET.SubElement(size, "height").text = str(image_np.shape[0])
            ET.SubElement(size, "depth").text = str(image_np.shape[2])
            
            
            for i in range(count):
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = category_index[classes[i]]['name']
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(np.asarray(boxes)[i][1] * image_np.shape[1]))
                ET.SubElement(bndbox, "ymin").text = str(int(np.asarray(boxes)[i][0] * image_np.shape[0]))
                ET.SubElement(bndbox, "xmax").text = str(int(np.asarray(boxes)[i][3] * image_np.shape[1]))
                ET.SubElement(bndbox, "ymax").text = str(int(np.asarray(boxes)[i][2] * image_np.shape[0]))
            
            tree = ET.ElementTree(annotation)
            tree.write(detected_output_dir + str(localtime) + '.xml')
            im = Image.fromarray(image_np)
            im.save(detected_output_dir + str(localtime) + '.jpg')
    #exit()


def find_overlap(boxes, scores, classes, current_frame_count, image_np, threshold, overlap_index, fps_video, overlap_output_dir, total_frame): #find overlap with two objects

    #print("image: ", image_np)
    #print("image.shape: ", image_np.shape)
    #print("==============================================")
    #print("current_frame_count: ", current_frame_count)
    #print("fps_video: ", fps_video)
    #print("current_video_time: ", current_frame_count / fps_video) 
    #print("total_video_time: ", total_frame / fps_video)
    overlap_index=overlap_index+1
    #print("boxes:\n ", np.squeeze(boxes))
    #print("scores:\n ", np.squeeze(scores))
    #print("classes:\n ", np.squeeze(classes).astype(np.int32))
    #print("index: ", index)


    #search person(class 1) and bicycle(class 2)
    master_boxes=[]
    slave_boxes=[]
    for i in range(100):
        if scores[0][i]>0.5:
            if classes[0][i]==3: #bicycle
                master_boxes.append(boxes[0][i])
            if classes[0][i]==1: #person
                slave_boxes.append(boxes[0][i])
        else:
            break
    master_boxes = np.asarray(master_boxes)
    slave_boxes = np.asarray(slave_boxes)

    #print("master_boxes: ", master_boxes)
    #print("slave_boxes: ", slave_boxes)
    
    for master_test in master_boxes:
        master_area = (master_test[2] - master_test[0]) * (master_test[3] - master_test[1])
        for slave_test in slave_boxes:
            overlap_X = (master_test[2] - master_test[0]) + (slave_test[2] - slave_test[0]) - (max(master_test[2], slave_test[2])- min(master_test[0], slave_test[0]))
            overlap_Y = (master_test[3] - master_test[1]) + (slave_test[3] - slave_test[1]) - (max(master_test[3], slave_test[3])- min(master_test[1], slave_test[1]))
            if overlap_X>=0 and overlap_Y>=0:
                overlap_area = overlap_X * overlap_Y
                ratio = overlap_area/master_area
                if ratio > threshold:
                    im = Image.fromarray(image_np)
                    im.save(overlap_output_dir + str(overlap_index) + '.jpg')
                    print("==============================================")
                    print("length and width: ", overlap_X, overlap_Y)
                    print("Area of overlap: ", overlap_X * overlap_Y * image_np.shape[0] * image_np.shape[1])
                    print("current_frame_count: ", current_frame_count)
                    print("fps_video: ", fps_video)
                    print("current_video_time: ", current_frame_count / fps_video) 
                    print("total_video_time: ", total_frame / fps_video)
                    print("==============================================\n")
    return overlap_index, fps_video


def get_boxes_distance_matrix(boxes, scores, classes, img_np, fps_video, score_criteria=0.5, displacement=False):
    """
    :boxes: of shape (1, 100, 4); (top, left, bottom, right)
    :scores: of shape (1, 100)
    :classes: of shape (1, 100)
    :score_criteria: score threshold
    :return: distance matrix for those boxes with score larger than score_criteria
    """
    # filter with score_criteria
    filtered_boxes = boxes[scores > score_criteria]

    num_boxes = filtered_boxes.shape[0]
    distance_matrix = np.zeros([num_boxes, num_boxes])
    for i in range(num_boxes):
        for j in range(i):  # halve iterations
            # save image for demo
            im = Image.fromarray(img_np)
            im.save('distance_' + str(fps_video) + '.jpg')
            # TODO It's worth discussed to define a better distance function.
            distance_matrix[i, j] = _get_two_boxes_boundary_distance(
                filtered_boxes[i], filtered_boxes[j], displacement)
            distance_matrix[j, i] = - distance_matrix[i, j] if displacement else distance_matrix[i, j]

            # demo
            a = filtered_boxes[i]
            b = filtered_boxes[j]
            print('focused on {}-th box {}\ncompare with {}-th box {}'.format(
                i, a, j, b))
            print('is_on_the_left: {}, is_on_the_right: {}, is_above: {}, is_below: {}'.format(
                is_box_on_the_left(a, b), is_box_on_the_right(a, b),
                is_box_above(a, b), is_box_below(a, b)
            ))

    return distance_matrix


def _get_two_boxes_boundary_distance(box_a, box_b, displacement=False):
    """
    :displacement: Displacement is distance with direction, so it's kinda a vector. If true, d(a, b) = - d(b, a).
    :return: (absolutely-)maximum displacement from one of its coordinate distance
    """
    box_a = np.reshape(box_a, -1)
    box_b = np.reshape(box_b, -1)

    overlap = is_x_overlapping(box_a, box_b) and is_y_overlapping(box_a, box_b)
    # if they overlap, distance = 0
    if overlap:
        return 0
    elif displacement:
        disp_x = box_a[3] - box_b[1] if abs(box_a[3] - box_b[1]) < abs(box_a[1] - box_b[3]) else box_a[1] - box_b[3]
        disp_y = box_a[2] - box_b[0] if abs(box_a[2] - box_b[0]) < abs(box_a[0] - box_b[2]) else box_a[0] - box_b[2]
        return disp_x if abs(disp_x) > abs(disp_y) else disp_y
    else:
        dist_x = min(abs(box_a[3] - box_b[1]), abs(box_a[1] - box_b[3]))
        dist_y = min(abs(box_a[2] - box_b[0]), abs(box_a[0] - box_b[2]))
        return max(dist_x, dist_y)


def is_x_overlapping(box_a, box_b):
    """
    overlap on X axis = l_b < l_a < r_b or la < l_b < r_a, similar for Y
    """
    return box_b[1] < box_a[1] < box_b[3] or box_a[1] < box_b[1] < box_a[3]


def is_y_overlapping(box_a, box_b):
    return box_b[0] < box_a[0] < box_b[2] or box_a[0] < box_b[0] < box_a[2]


def is_box_on_the_left(box_a, box_b):
    return box_b[3] < box_a[1] and not is_x_overlapping(box_a, box_b)


def is_box_on_the_right(box_a, box_b):
    return box_a[3] < box_b[1] and not is_x_overlapping(box_a, box_b)


def is_box_above(box_a, box_b):
    return box_b[2] < box_a[0] and not is_y_overlapping(box_a, box_b)


def is_box_below(box_a, box_b):
    return box_a[2] < box_b[0] and not is_y_overlapping(box_a, box_b)