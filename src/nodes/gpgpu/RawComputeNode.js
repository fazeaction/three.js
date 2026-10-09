import ComputeNode from './ComputeNode.js';
import { uniformGroup } from '../core/UniformGroupNode.js';
import { RawResourcesNode, planUniforms, planStorage, appendDeclarations } from '../code/RawShaderUtils.js';

/**
 * A compute node for a hand-written WGSL module, the compute counterpart of
 * {@link RawShaderNodeMaterial}.
 *
 * The author writes the complete module with `@compute @workgroup_size( ... ) fn main`. Storage
 * buffers, struct layouts and uniforms are declared in JS and referenced by name in WGSL; the
 * renderer appends their declarations after the author's text.
 *
 * ```js
 * const step = new RawComputeNode( {
 * 	source,                                       // WGSL with `@compute fn main`
 * 	count: particleCount,
 * 	workgroupSize: [ 64, 1, 1 ],
 * 	uniforms: { deltaT: { value: 0.04 } },        // -> uniforms.deltaT
 * 	structs: { Particle: { pos: 'vec2', vel: 'vec2' } },
 * 	storage: {
 * 		particlesA: { value: bufferA, type: 'Particle', access: 'read' },      // -> particlesA.value[ i ].pos
 * 		particlesB: { value: bufferB, type: 'Particle', access: 'read_write' },
 * 		cellCount: { value: counts, type: 'uint', access: 'read_write', atomic: true } // -> atomicAdd( &cellCount.value[ i ], 1u )
 * 	}
 * } );
 *
 * renderer.compute( step );
 * ```
 *
 * @augments ComputeNode
 */
class RawComputeNode extends ComputeNode {

	static get type() {

		return 'RawComputeNode';

	}

	/**
	 * Constructs a new raw compute node.
	 *
	 * @param {Object} parameters - The parameters.
	 * @param {string} parameters.source - The complete WGSL compute module.
	 * @param {number} parameters.count - The number of invocations to dispatch.
	 * @param {Array<number>} [parameters.workgroupSize=[64,1,1]] - The workgroup size, matching the module's `@workgroup_size`.
	 * @param {Object<string, {value: any, type: ?string}>} [parameters.uniforms={}] - Uniforms, referenced as `uniforms.<name>`.
	 * @param {Object<string, Object<string, string>>} [parameters.structs={}] - Struct layouts by name, usable as storage element types.
	 * @param {Object<string, {value: BufferAttribute, type: ?string, access: ?string, atomic: ?boolean}>} [parameters.storage={}] - Storage buffers by name; `type` is a key of `structs` or a type name used as is, `access` is `'read'` or `'read_write'`, `atomic` wraps the element type in `atomic<>`.
	 */
	constructor( { source, count, workgroupSize = [ 64, 1, 1 ], uniforms = {}, structs = {}, storage: buffers = {} } ) {

		const resources = new RawResourcesNode( ( builder ) => this._plan( builder ) );

		super( resources, [ ...workgroupSize, 1, 1 ].slice( 0, 3 ) );

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isRawComputeNode = true;

		/**
		 * Marks an object that owns its shader text; `NodeBuilder` dispatches `buildCode()` to it.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isRawShader = true;

		/**
		 * The complete WGSL compute module.
		 *
		 * @type {string}
		 */
		this.source = source;

		this.count = count;

		/**
		 * Uniforms as `{ name: { value, type? } }`.
		 *
		 * @type {Object<string, {value: any, type: ?string}>}
		 */
		this.uniforms = uniforms;

		/**
		 * Struct layouts by name.
		 *
		 * @type {Object<string, Object<string, string>>}
		 */
		this.structs = structs;

		/**
		 * Storage buffers by name.
		 *
		 * @type {Object<string, {value: BufferAttribute, type: ?string, access: ?string, atomic: ?boolean}>}
		 */
		this.buffers = buffers;

		this._group = uniformGroup( 'uniforms' );

	}

	generate( builder ) {

		// ComputeNode.generate() would also inject an early-return guard that registers a count
		// uniform; the author owns the entry point here, so only the resources are built.

		if ( builder.shaderStage === 'compute' ) this.computeNode.build( builder, 'void' );

	}

	/**
	 * Assembles the final module: the author's text followed by the generated declarations.
	 * Called by `NodeBuilder` in place of its own `buildCode()`.
	 *
	 * @param {NodeBuilder} builder - The current node builder.
	 */
	buildCode( builder ) {

		builder.sortBindingGroups();

		builder.computeShader = appendDeclarations( builder, 'compute', this.source, this.type );

	}

	_plan( /* builder */ ) {

		return [
			...planStorage( this.source, this.buffers, this.structs, this._group ),
			...planUniforms( this.source, this.uniforms, this._group, false )
		];

	}

}

export default RawComputeNode;
