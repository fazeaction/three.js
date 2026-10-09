import NodeMaterial from './NodeMaterial.js';
import { uniformGroup } from '../../nodes/core/UniformGroupNode.js';
import { NodeUpdateType } from '../../nodes/core/constants.js';
import { attribute } from '../../nodes/core/AttributeNode.js';
import { hashString } from '../../nodes/core/NodeUtils.js';
import { RawResourcesNode, parseVertexInputs, planUniforms, planStorage, appendDeclarations, inferUniformType } from '../../nodes/code/RawShaderUtils.js';

/**
 * A material for hand-written WGSL, the node system's counterpart of `RawShaderMaterial`.
 *
 * The author writes two complete WGSL modules, each with a `fn main` entry point. The renderer
 * appends the resource declarations it generated after the author's text, so pipelines, bind
 * groups, buffer uploads and the program cache stay the ordinary node-material ones.
 *
 * Three names derive from the `uniforms` dictionary keys: `uniforms.<key>` for plain values,
 * which share one uniform struct, and `<key>` plus `<key>_sampler` for textures. Renderer-provided
 * values such as `uniforms.modelViewMatrix` and `uniforms.projectionMatrix` are injected only when
 * the text mentions them. Vertex attributes are read from the `@location( n )` inputs of the
 * `@vertex` entry point, inline or through a struct, and must exist on the geometry. Storage
 * buffers declared in `storage` are read-only in the render stages and are referenced as
 * `<key>.value[ i ]`, with struct layouts from `structs` or a struct the author writes in WGSL.
 *
 * ```js
 * const material = new RawShaderNodeMaterial( {
 * 	vertexShader,   // WGSL with `@vertex fn main`
 * 	fragmentShader, // WGSL with `@fragment fn main`
 * 	uniforms: {
 * 		uTime: { value: 0 },
 * 		uMap: { value: texture }
 * 	}
 * } );
 * ```
 *
 * WebGPU backend only. No renderer output transforms, fog, clipping, skinning or morph targets
 * apply; the module is the author's.
 *
 * @augments NodeMaterial
 */
class RawShaderNodeMaterial extends NodeMaterial {

	static get type() {

		return 'RawShaderNodeMaterial';

	}

	/**
	 * Constructs a new raw shader node material.
	 *
	 * @param {Object} [parameters] - The configurable parameters of the material.
	 */
	constructor( parameters = {} ) {

		super();

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isRawShaderNodeMaterial = true;

		/**
		 * Marks an object that owns its shader text; `NodeBuilder` dispatches `buildCode()` to it.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isRawShader = true;

		/**
		 * The complete WGSL vertex module, with `@vertex fn main`.
		 *
		 * @type {string}
		 */
		this.vertexShader = '';

		/**
		 * The complete WGSL fragment module, with `@fragment fn main`.
		 *
		 * @type {string}
		 */
		this.fragmentShader = '';

		/**
		 * Uniforms as `{ name: { value, type? } }`. Values are read by reference every object.
		 *
		 * @type {Object<string, {value: any, type: ?string}>}
		 */
		this.uniforms = {};

		/**
		 * Optional explicit attribute names, index = `@location`. When `null`, the names are
		 * parsed from the vertex entry point.
		 *
		 * @type {?Array<string>}
		 * @default null
		 */
		this.attributes = null;

		/**
		 * Struct layouts by name, usable as storage element types.
		 *
		 * @type {Object<string, Object<string, string>>}
		 */
		this.structs = {};

		/**
		 * Storage buffers by name, read-only in the render stages. `type` is a key of
		 * {@link RawShaderNodeMaterial#structs} or a type name used as is.
		 *
		 * @type {Object<string, {value: BufferAttribute, type: ?string}>}
		 */
		this.storage = {};

		this._group = uniformGroup( 'uniforms', 1, NodeUpdateType.OBJECT ); // uploaded for every object, like the built-in object group

		this.vertexNode = new RawResourcesNode( ( builder ) => this._plan( builder, 'vertex' ) );
		this.fragmentNode = new RawResourcesNode( ( builder ) => this._plan( builder, 'fragment' ) );

		this.setValues( parameters );

	}

	setup( builder ) {

		if ( builder.renderer.backend.isWebGPUBackend !== true ) {

			throw new Error( 'RawShaderNodeMaterial: the WebGPU backend is required, the shaders are WGSL.' );

		}

		builder.addFlow( 'vertex', this.vertexNode );
		builder.addFlow( 'fragment', this.fragmentNode );

		builder.observer = this.setupObserver( builder );

	}

	/**
	 * Assembles the final modules: the author's text followed by the generated declarations.
	 * Called by `NodeBuilder` in place of its own `buildCode()`.
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 */
	buildCode( builder ) {

		builder.sortBindingGroups();

		builder.vertexShader = appendDeclarations( builder, 'vertex', this.vertexShader, this.type );
		builder.fragmentShader = appendDeclarations( builder, 'fragment', this.fragmentShader, this.type );

	}

	_plan( builder, stage ) {

		const source = stage === 'vertex' ? this.vertexShader : this.fragmentShader;
		const nodes = [];

		if ( stage === 'vertex' ) {

			for ( const name of this.attributes || parseVertexInputs( source ) ) {

				if ( builder.hasGeometryAttribute( name ) !== true ) {

					throw new Error( `RawShaderNodeMaterial: the geometry has no attribute "${ name }", which the vertex shader declares.` );

				}

				nodes.push( attribute( name ) );

			}

		}

		nodes.push( ...planStorage( source, this.storage, this.structs, this._group ) );
		nodes.push( ...planUniforms( source, this.uniforms, this._group, true ) );

		return nodes;

	}

	customProgramCacheKey() {

		const uniforms = Object.entries( this.uniforms ).map( ( [ name, entry ] ) => `${ name }:${ entry.type || inferUniformType( entry.value ) }` ).join( ',' );
		const buffers = Object.entries( this.storage ).map( ( [ name, entry ] ) => `${ name }:${ entry.type || '' }` ).join( ',' );
		const structs = JSON.stringify( this.structs );

		return this.type + hashString( this.vertexShader ) + hashString( this.fragmentShader ) + hashString( uniforms + '|' + buffers + '|' + structs );

	}

}

export default RawShaderNodeMaterial;
