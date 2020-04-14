# ##### BEGIN GPL LICENSE BLOCK #####

# Blendermada client.
# Add-on for Blender 3D to access content from http://blendermada.com
# Copyright (C) 2014  Sergey Ozerov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ##### END GPL LICENSE BLOCK #####


bl_info = {
    "name": "Blendermada Client",
    "author": "Sergey Ozerov, <ozzyrov@gmail.com>",
    "version": (2, 0, 1),
    "blender": (2, 80, 0),
    "location": "Properties, Material, Blendermada Client",
    "description": "Browse and download materials from online material database.",
    "warning": "Beta version",
    "wiki_url": "http://blendermada.com/addon/",
    "tracker_url": "https://github.com/TrueCryer/blendermada_client/issues",
    "category": "Material",
}

from threading import Lock

import bpy
import bgl
from bgl import Buffer as Buffer
from bpy.props import *
import gpu
from gpu_extras.batch import batch_for_shader


from urllib import request, parse
import json
import os
import pickle
import time
from datetime import datetime


ENGINE_MAPPING = {
    'BLENDER_RENDER': 'int',
    'BLENDER_GAME': 'int',
    'CYCLES': 'cyc',
    'EEVEE': 'eve',
}


########################################################################
########################################################################


#import gpu
#from gpu_extras.batch import batch_for_shader

GL_LINES = 0
GL_LINE_STRIP = 1
GL_LINE_LOOP = 2
GL_TRIANGLES = 5
GL_TRIANGLE_FAN = 6
GL_QUADS = 4

class InternalData:
    __inst = None
    __lock = Lock()

    def __init__(self):
        raise NotImplementedError("Not allowed to call constructor")

    @classmethod
    def __internal_new(cls):
        inst = super().__new__(cls)
        inst.color = [1.0, 1.0, 1.0, 1.0]
        inst.line_width = 1.0

        return inst

    @classmethod
    def get_instance(cls):
        if not cls.__inst:
            with cls.__lock:
                if not cls.__inst:
                    cls.__inst = cls.__internal_new()

        return cls.__inst

    def init(self):
        self.clear()

    def set_prim_mode(self, mode):
        self.prim_mode = mode

    def set_dims(self, dims):
        self.dims = dims

    def add_vert(self, v):
        self.verts.append(v)

    def add_tex_coord(self, uv):
        self.tex_coords.append(uv)

    def set_color(self, c):
        self.color = c

    def set_line_width(self, width):
        self.line_width = width

    def clear(self):
        self.prim_mode = None
        self.verts = []
        self.dims = None
        self.tex_coords = []

    def get_verts(self):
        return self.verts

    def get_dims(self):
        return self.dims

    def get_prim_mode(self):
        return self.prim_mode

    def get_color(self):
        return self.color

    def get_line_width(self):
        return self.line_width

    def get_tex_coords(self):
        return self.tex_coords


def glColor4f(r, g, b, a):
    inst = InternalData.get_instance()
    inst.set_color([r, g, b, a])


def glBegin(mode):
    inst = InternalData.get_instance()
    inst.init()
    inst.set_prim_mode(mode)


def _get_transparency_shader():
    vertex_shader = '''
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;

    in vec2 pos;
    in vec2 texCoord;
    out vec2 uvInterp;

    void main()
    {
        uvInterp = texCoord;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos.xy, 0.0, 1.0);
        gl_Position.z = 1.0;
    }
    '''

    fragment_shader = '''
    uniform sampler2D image;
    uniform vec4 color;

    in vec2 uvInterp;
    out vec4 fragColor;

    void main()
    {
        fragColor = texture(image, uvInterp);
        fragColor.a = color.a;
    }
    '''

    return vertex_shader, fragment_shader


def glEnd():
    inst = InternalData.get_instance()

    color = inst.get_color()
    coords = inst.get_verts()
    tex_coords = inst.get_tex_coords()
    if inst.get_dims() == 2:
        if len(tex_coords) == 0:
            shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        else:
            #shader = gpu.shader.from_builtin('2D_IMAGE')
            vert_shader, frag_shader = _get_transparency_shader()
            shader = gpu.types.GPUShader(vert_shader, frag_shader)
    else:
        raise NotImplemented("get_dims() != 2")

    if len(tex_coords) == 0:
        data = {
            "pos": coords,
        }
    else:
        data = {
            "pos": coords,
            "texCoord": tex_coords
        }

    if inst.get_prim_mode() == GL_LINES:
        indices = []
        for i in range(0, len(coords), 2):
            indices.append([i, i + 1])
        batch = batch_for_shader(shader, 'LINES', data, indices=indices)

    elif inst.get_prim_mode() == GL_LINE_STRIP:
        batch = batch_for_shader(shader, 'LINE_STRIP', data)


    elif inst.get_prim_mode() == GL_LINE_LOOP:
        data["pos"].append(data["pos"][0])
        batch = batch_for_shader(shader, 'LINE_STRIP', data)

    elif inst.get_prim_mode() == GL_TRIANGLES:
        indices = []
        for i in range(0, len(coords), 3):
            indices.append([i, i + 1, i + 2])
        batch = batch_for_shader(shader, 'TRIS', data, indices=indices)

    elif inst.get_prim_mode() == GL_TRIANGLE_FAN:
        indices = []
        for i in range(1, len(coords) - 1):
            indices.append([0, i, i + 1])
        batch = batch_for_shader(shader, 'TRIS', data, indices=indices)

    elif inst.get_prim_mode() == GL_QUADS:
        indices = []
        for i in range(0, len(coords), 4):
            indices.extend([[i, i + 1, i + 2], [i + 2, i + 3, i]])
        batch = batch_for_shader(shader, 'TRIS', data, indices=indices)
    else:
        raise NotImplemented("get_prim_mode() != (GL_LINES|GL_TRIANGLES|GL_QUADS)")

    shader.bind()
    if len(tex_coords) != 0:
        shader.uniform_float("modelViewMatrix", gpu.matrix.get_model_view_matrix())
        shader.uniform_float("projectionMatrix", gpu.matrix.get_projection_matrix())
        shader.uniform_int("image", 0)
    shader.uniform_float("color", color)
    batch.draw(shader)

    inst.clear()


def glVertex2f(x, y):
    inst = InternalData.get_instance()
    inst.add_vert([x, y])
    inst.set_dims(2)


def glTexCoord2f(u, v):
    inst = InternalData.get_instance()
    inst.add_tex_coord([u, v])


########################################################################
########################################################################


def get_cache_path():
    path = bpy.context.preferences.addons[__name__].preferences.cache_path
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, 'bmd_cache')
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def dump_data(data, filepath):
    with open(filepath, 'wb+') as f:
        pickle.dump(data, f)

def load_data(filepath):
    with open(filepath, 'rb+') as f:
        data = pickle.load(f)
    return data

def file_expired(filepath, seconds_to_live):
    if os.path.exists(filepath):
        if time.mktime(datetime.now().timetuple()) - os.path.getmtime(filepath) < seconds_to_live:
            return False
    return True

########################################################################
########################################################################


def get_engine():
    engine = bpy.context.scene.render.engine
    try:
        return ENGINE_MAPPING[engine]
    except:
        return ''

def get_proxy_handlers():
    handlers = []
    addon_prefs = bpy.context.preferences.addons[__name__].preferences
    if addon_prefs.proxy_use_proxy:
        proxy_handler = request.ProxyHandler({
            'http': '{host}:{port}'.format(
                host=addon_prefs.proxy_server,
                port=addon_prefs.proxy_port,
            ),
        })
        handlers.append(proxy_handler)
        if addon_prefs.proxy_use_auth:
            pass
    return handlers

def bmd_urlopen(url, **kwargs):
    full_url = parse.urljoin('http://blendermada.com/', url)
    params = parse.urlencode(kwargs)
    handlers = get_proxy_handlers()
    opener = request.build_opener(*handlers)
    request.install_opener(opener)
    return request.urlopen('%s?%s' % (full_url, params))

def get_materials(category):
    engine = get_engine()
    filepath = os.path.join(get_cache_path(), '{}-cat-{}'.format(engine, category))
    if file_expired(filepath, 300):
        r = bmd_urlopen(
            '/api/materials/materials.json',
            engine=engine,
            category=category,
        )
        ans = json.loads(str(r.read(), 'UTF-8'))
        dump_data(ans, filepath)
    return load_data(filepath)

def get_favorites():
    engine = get_engine()
    filepath = os.path.join(get_cache_path(), '{}-cat-fav'.format(engine))
    if file_expired(filepath, 300):
        addon_prefs = bpy.context.preferences.addons[__name__]
        r = bmd_urlopen(
            '/api/materials/v1/favorites.json',
            engine=engine,
            key=addon_prefs.preferences.api_key,
        )
        ans = json.loads(str(r.read(), 'UTF-8'))
        dump_data(ans, filepath)
    return load_data(filepath)

def get_categories():
    filepath = os.path.join(get_cache_path(), 'categories')
    if file_expired(filepath, 300):
        r = bmd_urlopen('/api/materials/categories.json')
        ans = json.loads(str(r.read(), 'UTF-8'))
        dump_data(ans, filepath)
    return load_data(filepath)

def get_material_detail(id):
    filepath = os.path.join(get_cache_path(), 'mat-%s' % (id,))
    if file_expired(filepath, 300):
        r = bmd_urlopen('/api/materials/material.json', id=id)
        ans = json.loads(str(r.read(), 'UTF-8'))
        dump_data(ans, filepath)
    return load_data(filepath)

def get_image(url):
    filepath = os.path.join(get_cache_path(), 'images')
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    filepath = os.path.join(filepath, url.split('/')[-1])
    if file_expired(filepath, 300):
        r = bmd_urlopen(url)
        with open(filepath, 'wb+') as f:
            f.write(r.read())
    return filepath

def get_library(url):
    filepath = os.path.join(get_cache_path(), 'files')
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    filepath = os.path.join(filepath, url.split('/')[-1])
    if file_expired(filepath, 300):
        r = bmd_urlopen(url)
        with open(filepath, 'wb+') as f:
            f.write(r.read())
    return filepath

########################################################################
########################################################################


class Preview(object):

    def __init__(self):

        self.activated = False

        self.x = 10
        self.y = 10
        try:
            addon_prefs = bpy.context.preferences.addons[__name__]
        except KeyError:
            self.set_preview_size(False)
        else:
            self.set_preview_size(addon_prefs.preferences.use_big_preview)

        self.move = False
        self.glImage = None
        self.bindcode = None

    def set_preview_size(self, big_preview):
        if big_preview:
            self.width, self.height = 256, 256
        else:
            self.width, self.height = 128, 128

    def load_image(self, image_url):
        self.glImage = bpy.data.images.load(get_image(image_url))
        self.glImage.gl_load(frame=bgl.GL_NEAREST) #, bgl.GL_NEAREST)
        #if bpy.app.version < (2, 77):
        self.bindcode = self.glImage.bindcode
        #else:
            #self.bindcode = self.glImage.bindcode[0]

    def unload_image(self):
        if not self.glImage == None:
            self.glImage.gl_free()
            self.glImage.user_clear()
            bpy.data.images.remove(self.glImage)
        self.glImage = None
        self.bindcode = None

    def activate(self, context):
        self.handler = bpy.types.SpaceProperties.draw_handler_add(
                                   render_callback,
                                   (self, context), 'WINDOW', 'POST_PIXEL')
        bpy.context.scene.cursor.location.x += 0.0 # refresh display
        self.activated = True

    def deactivate(self, context):
        bpy.types.SpaceProperties.draw_handler_remove(self.handler, 'WINDOW')
        bpy.context.scene.cursor.location.x += 0.0 # refresh display
        self.activated = False

    def event_callback(self, context, event):
        if self.activated == False:
            return {'FINISHED'}
        if event.type == 'MIDDLEMOUSE':
            if event.value == 'PRESS':
                self.move = True
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                self.move = False
                return {'RUNNING_MODAL'}
        if self.move and event.type == 'MOUSEMOVE':
            self.x += event.mouse_x - event.mouse_prev_x
            self.y += event.mouse_y - event.mouse_prev_y
            bpy.context.scene.cursor.location.x += 0.0 # refresh display
            return {'RUNNING_MODAL'}
        else:
            return {'PASS_THROUGH'}


def image_changed(self, value):
    bmd_preview.unload_image()
    if value:
        bmd_preview.load_image(value)


def render_callback(self, context):
    if self.bindcode != None:
		
        bgl.glEnable(bgl.GL_BLEND)
        bgl.glEnable(bgl.GL_TEXTURE_2D)
        bgl.glActiveTexture(bgl.GL_TEXTURE0)
        bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.bindcode)
        
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(self.x, self.y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(self.x + self.width, self.y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(self.x + self.width, self.y + self.height)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(self.x, self.y + self.height)
        glEnd()


########################################################################
########################################################################


def update_categories(context):
    context.scene.bmd_category_list.clear()
    addon_prefs = bpy.context.preferences.addons[__name__]
    if addon_prefs.preferences.api_key != '':
        a = context.scene.bmd_category_list.add()
        a.id   = 0
        a.slug = 'favorites'
        a.name = '<Favorites>'
    for i in get_categories():
        a = context.scene.bmd_category_list.add()
        a.id   = i['id']
        a.slug = i['slug']
        a.name = i['name']
    update_materials(None, context)

def update_materials(self, context):
    context.scene.bmd_material_list.clear()
    id = context.scene.bmd_category_list[context.scene.bmd_category_list_idx].id
    if id == 0: # Favorites
        mats = get_favorites()
    else:
        mats = get_materials(id)
    for i in mats:
        a = context.scene.bmd_material_list.add()
        a.id = i['id']
        a.slug = i['slug']
        a.name = i['name']
    if len(context.scene.bmd_material_list) > 0:
        context.scene.bmd_material_list_idx = 0
        update_active_material(self, context)

def update_active_material(self, context):
    mat = get_material_detail(
        context.scene.bmd_material_list[context.scene.bmd_material_list_idx].id,
    )
    context.scene.bmd_material_active.id = mat['id']
    context.scene.bmd_material_active.slug = mat['slug']
    context.scene.bmd_material_active.name = mat['name']
    context.scene.bmd_material_active.description = mat['description']
    context.scene.bmd_material_active.downloads = mat['downloads']
    context.scene.bmd_material_active.rating = mat['rating']
    context.scene.bmd_material_active.votes = mat['votes']
    context.scene.bmd_material_active.storage_name = mat['storage_name']
    context.scene.bmd_material_active.image_url = mat['image']
    context.scene.bmd_material_active.library_url = mat['storage']

########################################################################
########################################################################


class BMDCategoryPG(bpy.types.PropertyGroup):
    id : IntProperty()
    slug : StringProperty()
    name : StringProperty()

bpy.utils.register_class(BMDCategoryPG)
bpy.types.Scene.bmd_category_list = CollectionProperty(type=BMDCategoryPG)
bpy.types.Scene.bmd_category_list_idx = IntProperty(update=update_materials)
bpy.types.Scene.bmd_category_active = PointerProperty(type=BMDCategoryPG)


class BMDMaterialListPG(bpy.types.PropertyGroup):
    id : IntProperty()
    slug : StringProperty()
    name : StringProperty()

bpy.utils.register_class(BMDMaterialListPG)
bpy.types.Scene.bmd_material_list = CollectionProperty(type=BMDMaterialListPG)
bpy.types.Scene.bmd_material_list_idx = IntProperty(update=update_active_material)


class BMDMaterialDetailPG(bpy.types.PropertyGroup):
    id : IntProperty()
    slug : StringProperty()
    name : StringProperty()
    description : StringProperty()
    downloads : IntProperty()
    rating : FloatProperty()
    votes : IntProperty()
    description : StringProperty()
    storage_name : StringProperty()
    image_url : StringProperty(set=image_changed)
    library_url : StringProperty()

bpy.utils.register_class(BMDMaterialDetailPG)
bpy.types.Scene.bmd_material_active = PointerProperty(type=BMDMaterialDetailPG)
bpy.types.Scene.bmd_preview = IntProperty()

########################################################################
########################################################################


def material_imported(context):
    name = context.scene.bmd_material_active.storage_name
    if name in bpy.data.materials.keys():
        return True
    return False

########################################################################
########################################################################


class BMD_PT_Panel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Blendermada Client"
    bl_idname = "BMD_PT_ClientPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        row = layout.row(align=True)
        row.operator('bmd.update', icon="FILE_REFRESH")
        row.operator('bmd.preview', icon="MATERIAL")
        row.operator('bmd.import', icon="IMPORT")
        row.separator()
        row.operator('bmd.help', icon="HELP", text="")
        row.operator('bmd.support', icon="SOLO_ON", text="")
        row = layout.row()
        col = row.column()
        col.label(text='Category')
        col.template_list('BMD_UL_CategoryList', '', context.scene, 'bmd_category_list', context.scene, 'bmd_category_list_idx', rows=6)
        col = row.column()
        col.label(text='Material')
        col.template_list('BMD_UL_MaterialList', '', context.scene, 'bmd_material_list', context.scene, 'bmd_material_list_idx', rows=6)
        layout.label(text='Material Detail')
        box = layout.box()
        row = box.row()
        col = row.column()
        col.label(text=context.scene.bmd_material_active.name)
        col = row.column()
        col.label(text=': {}'.format(context.scene.bmd_material_active.downloads), icon="IMPORT")
        col.label(text=': {:1.2f} ({} votes)'.format(context.scene.bmd_material_active.rating, context.scene.bmd_material_active.votes), icon="SOLO_ON")
        for row in context.scene.bmd_material_active.description.split('\n'):
            box.label(text=row)
        #layout.template_image(context.scene, 'bmd_preview', {'NULL'})


class BMD_UL_MaterialList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name)


class BMD_UL_CategoryList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=item.name)


class BMDImport(bpy.types.Operator):
    bl_idname = "bmd.import"
    bl_label = "Import"
    def execute(self, context):
        if material_imported(context):
            self.report(
                {'WARNING'},
                'Material with name \'%s\' already exists here. Please, rename it to import new one.' % (
                    context.scene.bmd_material_active.storage_name,
                )
            )
            return {'CANCELLED'}
        else:
            storage = get_library(context.scene.bmd_material_active.library_url)
            directory = os.path.join(storage, 'Material', '')
            filename = context.scene.bmd_material_active.storage_name
            if bpy.app.version < (2, 72):
                bpy.ops.wm.link_append(
                    filepath=('%s%s' % (directory, filename)),
                    directory=directory,
                    filename=filename,
                    relative_path=True,
                    link=False,
                )
            else:
                bpy.ops.wm.append(
                    filepath=('%s%s' % (directory, filename)),
                    directory=directory,
                    filename=filename,
                )
            if not material_imported(context): # some error while importing
                self.report(
                    {'WARNING'},
                    'Material cannot be imported. Maybe library has been damaged. Please, report about it to Blendermada administrator.',
                )
                return {'CANCELLED'}
            else:
                ao = bpy.context.active_object
                if hasattr(ao.data, 'materials'): # object isn't lamp or camera
                    if len(ao.data.materials) == 0: # there is no material slot in object
                        ao.data.materials.append(bpy.data.materials[context.scene.bmd_material_active.storage_name])
                    else:
                        ao.material_slots[ao.active_material_index].material = bpy.data.materials[context.scene.bmd_material_active.storage_name]
                self.report({'INFO'}, 'Material was imported succesfully.')
                return {'FINISHED'}


class BMDUpdate(bpy.types.Operator):
    bl_idname = 'bmd.update'
    bl_label = 'Update'

    def execute(self, context):
        update_categories(context)
        return {'FINISHED'}


class BMDPreview(bpy.types.Operator):
    bl_idname = 'bmd.preview'
    bl_label = 'Preview'
    def __init__(self):
        super(BMDPreview, self).__init__()
    def modal(self, context, event):
        return bmd_preview.event_callback(context, event)
    def invoke(self, context, event):
        if not bmd_preview.activated:
            bmd_preview.activate(context)
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            bmd_preview.deactivate(context)
            return {'FINISHED'}


class BMDHelp(bpy.types.Operator):
    bl_idname = 'bmd.help'
    bl_label = "Help"
    bl_description = "View online help"

    def execute(self, context):
        bpy.ops.wm.url_open(url="http://blendermada.com/addon/")
        return {'FINISHED'}


class BMDSupport(bpy.types.Operator):
    bl_idname = 'bmd.support'
    bl_label = "Support"
    bl_description = "Support Blendermada"

    def execute(self, context):
        bpy.ops.wm.url_open(url="http://blendermada.com/about/")
        return {'FINISHED'}


def preview_size_update(self, context):
    addon_prefs = context.preferences.addons[__name__].preferences
    bmd_preview.set_preview_size(addon_prefs.use_big_preview)


class BMDAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    cache_path: StringProperty(
        name="Specific cache directory",
        subtype='DIR_PATH',
        description="Change this path if you have some problems with cache saving",
        default=os.path.expanduser(os.path.join('~', '.blendermada')),
    )
    use_big_preview: BoolProperty(
        name="Use a big preview",
        description="Use 256x256 previews instead of 128x128",
        update=preview_size_update,
    )
    api_key: StringProperty(
        name="API key",
        description="Use it to access your favorites materials",
    )
    proxy_use_proxy: BoolProperty(
        name="Use proxy",
        description="Use proxy for requests",
    )
    proxy_server: StringProperty(
        name="Server",
    )
    proxy_port: StringProperty(
        name="Port",
    )
    proxy_use_auth: BoolProperty(
        name="Use proxy authentication",
    )
    proxy_user: StringProperty(
        name="User",
    )
    proxy_password: StringProperty(
        name="Password",
    )
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "use_big_preview")
        layout.prop(self, "cache_path")
        layout.separator()
        layout.label(text="Authentication")
        layout.prop(self, "api_key")
        layout.separator()
        layout.label(text="Proxy")
        row = layout.row()
        row.prop(self, "proxy_use_proxy")
        if self.proxy_use_proxy:
            row.prop(self, "proxy_server")
            row.prop(self, "proxy_port")
            row = layout.row()
            row.prop(self, "proxy_use_auth")
            if self.proxy_use_auth:
                row.prop(self, "proxy_user")
                row.prop(self, "proxy_password")


def register():
    bpy.utils.register_class(BMD_PT_Panel)
    bpy.utils.register_class(BMDImport)
    bpy.utils.register_class(BMD_UL_MaterialList)
    bpy.utils.register_class(BMD_UL_CategoryList)
    bpy.utils.register_class(BMDUpdate)
    bpy.utils.register_class(BMDPreview)
    bpy.utils.register_class(BMDHelp)
    bpy.utils.register_class(BMDSupport)
    bpy.utils.register_class(BMDAddonPreferences)

    global bmd_preview
    bmd_preview = Preview()


def unregister():
    bpy.utils.unregister_class(BMD_PT_Panel)
    bpy.utils.unregister_class(BMDImport)
    bpy.utils.unregister_class(BMD_UL_MaterialList)
    bpy.utils.unregister_class(BMD_UL_CategoryList)
    bpy.utils.unregister_class(BMDUpdate)
    bpy.utils.unregister_class(BMDPreview)
    bpy.utils.unregister_class(BMDHelp)
    bpy.utils.unregister_class(BMDSupport)
    bpy.utils.unregister_class(BMDAddonPreferences)


if __name__ == '__main__':
    register()
