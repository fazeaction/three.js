import Node from '../core/Node.js';
import { uniform } from '../core/UniformNode.js';
import { reference } from '../accessors/ReferenceNode.js';
import { storage } from '../accessors/StorageBufferNode.js';
import StructTypeNode from '../core/StructTypeNode.js';
import { getValueType } from '../core/NodeUtils.js';
import { Matrix3 } from '../../math/Matrix3.js';
import { Matrix4 } from '../../math/Matrix4.js';
import { Vector3 } from '../../math/Vector3.js';

// Shared machinery for hand-written shader modules, see RawShaderNodeMaterial and RawComputeNode.
//
// The author writes the complete module, entry points included. The builder still owns the
// resource declarations, because WebGPU has no program reflection and the bind group layout and
// vertex buffer slots must be known before a pipeline exists. Those declarations are generated
// from the JS side and appended after the author's text. WGSL module-scope declarations are order
// independent, so the author's line numbers survive in compiler diagnostics.

const _mv = /*@__PURE__*/ new Matrix4();

/**
 * Renderer-provided values. A raw shader gets one of these only when its text mentions
 * `uniforms.<name>`, the way the WebGL raw shader path uploaded standard uniforms by name.
 *
 * @private
 * @type {Object<string, function(): UniformNode>}
 */
const PROVIDED = {
	modelMatrix: () => uniform( new Matrix4() ).onObjectUpdate( ( { object }, self ) => self.value.copy( object.matrixWorld ) ),
	modelViewMatrix: () => uniform( new Matrix4() ).onObjectUpdate( ( { object, camera }, self ) => self.value.multiplyMatrices( camera.matrixWorldInverse, object.matrixWorld ) ),
	normalMatrix: () => uniform( new Matrix3() ).onObjectUpdate( ( { object, camera }, self ) => self.value.getNormalMatrix( _mv.multiplyMatrices( camera.matrixWorldInverse, object.matrixWorld ) ) ),
	projectionMatrix: () => uniform( new Matrix4() ).onRenderUpdate( ( { camera }, self ) => self.value.copy( camera.projectionMatrix ) ),
	viewMatrix: () => uniform( new Matrix4() ).onRenderUpdate( ( { camera }, self ) => self.value.copy( camera.matrixWorldInverse ) ),
	cameraPosition: () => uniform( new Vector3() ).onRenderUpdate( ( { camera }, self ) => self.value.setFromMatrixPosition( camera.matrixWorld ) )
};

/**
 * Removes comments, which never declare or reference anything.
 *
 * @param {string} source - Shader source.
 * @return {string} The source without comments.
 */
export function stripComments( source ) {

	return source.replace( /\/\*[\s\S]*?\*\//g, '' ).replace( /\/\/[^\n]*/g, '' );

}

/**
 * Returns a regular expression matching `name` as a bare identifier, not preceded by a word
 * character or a dot.
 *
 * @param {string} name - The identifier.
 * @return {RegExp} The expression.
 */
export function identifierRegExp( name ) {

	return new RegExp( `(?<![\\w.])${ name }\\b` );

}

/**
 * Infers the node type of a uniform value.
 *
 * @param {any} value - The uniform value.
 * @return {string} The node type.
 */
export function inferUniformType( value ) {

	if ( value && value.isTexture === true ) return value.isCubeTexture === true ? 'cubeTexture' : 'texture';

	const type = getValueType( value );

	if ( type === null || type === 'node' || type === 'string' || type === 'shader' ) {

		throw new Error( 'RawShader: cannot infer the type of a uniform value; set "type" explicitly.' );

	}

	return type;

}

function splitTopLevel( text ) {

	// split on commas outside <> and ()

	const parts = [];
	let depth = 0, current = '';

	for ( const ch of text ) {

		if ( ch === '<' || ch === '(' ) depth ++;
		else if ( ch === '>' || ch === ')' ) depth --;

		if ( ch === ',' && depth === 0 ) {

			parts.push( current );
			current = '';

		} else {

			current += ch;

		}

	}

	parts.push( current );

	return parts;

}

/**
 * Reads the `@location( n )` inputs of the `@vertex` entry point, declared inline or through
 * struct parameters, and returns the attribute names ordered by location. Locations must be
 * dense from 0 because the renderer assigns `shaderLocation` by array index.
 *
 * @param {string} source - The WGSL vertex module.
 * @return {Array<string>} The geometry attribute names, index = location.
 */
export function parseVertexInputs( source ) {

	source = stripComments( source );

	const entry = /@vertex[\s\S]*?\bfn\s+\w+\s*\(([\s\S]*?)\)\s*(?:->|\{)/.exec( source );

	if ( entry === null ) throw new Error( 'RawShaderNodeMaterial: vertexShader has no @vertex entry point.' );

	const inputs = [];

	const collect = ( paramList ) => {

		for ( const raw of splitTopLevel( paramList ) ) {

			const param = raw.trim();

			if ( param === '' ) continue;

			const match = /^((?:@\w+\s*(?:\([^)]*\))?\s*)*)(\w+)\s*:\s*([\w<>,\s]+?)\s*$/.exec( param );

			if ( match === null ) throw new Error( `RawShaderNodeMaterial: cannot read vertex input "${ param }".` );

			const attributes = match[ 1 ], name = match[ 2 ], type = match[ 3 ].trim();
			const location = /@location\s*\(\s*(\d+)\s*\)/.exec( attributes );

			if ( location !== null ) {

				inputs.push( { location: Number( location[ 1 ] ), name } );

			} else if ( /@builtin/.test( attributes ) ) {

				continue;

			} else {

				const struct = new RegExp( `struct\\s+${ type }\\s*\\{([\\s\\S]*?)\\}` ).exec( source );

				if ( struct === null ) throw new Error( `RawShaderNodeMaterial: vertex input "${ name }" has type "${ type }", which is neither @location nor a struct declared in the shader.` );

				collect( struct[ 1 ] );

			}

		}

	};

	collect( entry[ 1 ] );

	inputs.sort( ( a, b ) => a.location - b.location );

	inputs.forEach( ( input, index ) => {

		if ( input.location !== index ) throw new Error( `RawShaderNodeMaterial: vertex @location values must be dense from 0; found ${ inputs.map( i => i.location ).join( ', ' ) }.` );

	} );

	return inputs.map( input => input.name );

}

function setGroup( node, group ) {

	if ( typeof node.setGroup === 'function' ) node.setGroup( group );

	return node;

}

/**
 * Creates the nodes for every `uniforms.<name>` the source references and every texture entry
 * the source mentions by bare name. Dictionary entries use `{ value, type? }` and are read by
 * reference every object, exactly like `ShaderMaterial` uniforms.
 *
 * @param {string} source - Shader source of one stage.
 * @param {Object<string, {value: any, type: ?string}>} uniforms - The uniforms dictionary.
 * @param {UniformGroupNode} group - The uniform group that gives the `uniforms.` prefix.
 * @param {boolean} allowProvided - Whether renderer-provided values may be injected.
 * @return {Array<Node>} The resource nodes.
 */
export function planUniforms( source, uniforms, group, allowProvided ) {

	source = stripComments( source );

	const nodes = [];
	const used = new Set();

	for ( const match of source.matchAll( /\buniforms\.(\w+)/g ) ) used.add( match[ 1 ] );

	for ( const name of used ) {

		const entry = uniforms[ name ];
		let node;

		if ( entry !== undefined ) {

			node = reference( 'value', entry.type || inferUniformType( entry.value ), entry );

		} else if ( allowProvided === true && PROVIDED[ name ] !== undefined ) {

			node = PROVIDED[ name ]();

		} else {

			const provided = allowProvided === true ? ' Renderer-provided: ' + Object.keys( PROVIDED ).join( ', ' ) + '.' : '';

			throw new Error( `RawShader: "uniforms.${ name }" is not declared. Declared: ${ Object.keys( uniforms ).join( ', ' ) || 'none' }.${ provided }` );

		}

		nodes.push( setGroup( node.setName( name ), group ) );

	}

	for ( const [ name, entry ] of Object.entries( uniforms ) ) {

		if ( entry.value && entry.value.isTexture === true && identifierRegExp( name ).test( source ) ) {

			nodes.push( setGroup( reference( 'value', inferUniformType( entry.value ), entry ).setName( name ), group ) );

		}

	}

	return nodes;

}

/**
 * Creates the nodes for every declared storage buffer the source mentions by bare name.
 * A buffer's `type` is either a key of `structs`, which the builder then emits as a WGSL struct,
 * or a type name used as is, so the author may declare the struct in WGSL instead.
 *
 * @param {string} source - Shader source of one stage.
 * @param {Object<string, {value: BufferAttribute, type: ?string, access: ?string, atomic: ?boolean}>} buffers - Storage buffers by name.
 * @param {Object<string, Object<string, string>>} structs - Struct layouts by name.
 * @param {UniformGroupNode} group - The uniform group, one bind group for every resource.
 * @return {Array<Node>} The resource nodes.
 */
export function planStorage( source, buffers, structs, group ) {

	source = stripComments( source );

	const nodes = [];
	const structTypes = {};

	for ( const [ name, layout ] of Object.entries( structs ) ) {

		structTypes[ name ] = new StructTypeNode( layout, name );

	}

	for ( const [ name, declaration ] of Object.entries( buffers ) ) {

		if ( identifierRegExp( name ).test( source ) !== true ) continue;

		const type = declaration.type !== undefined ? ( structTypes[ declaration.type ] || declaration.type ) : null;
		const node = storage( declaration.value, type );

		if ( declaration.access === 'read' ) node.toReadOnly();
		if ( declaration.atomic === true ) node.setAtomic( true );

		nodes.push( setGroup( node.setName( name ), group ) );

	}

	return nodes;

}

/**
 * Returns the author's module followed by the declarations the builder generated for the
 * given stage.
 *
 * @param {NodeBuilder} builder - The builder, after `sortBindingGroups()`.
 * @param {('vertex'|'fragment'|'compute')} stage - The shader stage.
 * @param {string} source - The author's module.
 * @param {string} owner - Name used in the marker comment.
 * @return {string} The complete module.
 */
export function appendDeclarations( builder, stage, source, owner ) {

	builder.setShaderStage( stage );

	const structs = builder.getStructs( stage );
	const uniforms = builder.getUniforms( stage );

	builder.setShaderStage( null );

	return `${ source.trimEnd() }\n\n// ---- declarations generated by ${ owner } ----\n${ structs }\n${ uniforms }\n${ builder.getSignature() }`;

}

/**
 * A flow node that registers hand-written shader resources with the builder.
 *
 * One instance per shader stage. Its `generate()` builds every planned resource node in
 * declaration order, which registers the bindings and attributes, and emits no code of its own.
 *
 * @augments Node
 */
class RawResourcesNode extends Node {

	static get type() {

		return 'RawResourcesNode';

	}

	/**
	 * Constructs a new resources node.
	 *
	 * @param {function(NodeBuilder): Array<Node>} plan - Returns the resource nodes for the current stage.
	 */
	constructor( plan ) {

		super( 'void' );

		/**
		 * Returns the resource nodes for the current stage.
		 *
		 * @type {function(NodeBuilder): Array<Node>}
		 */
		this.plan = plan;

		/**
		 * The planned resource nodes.
		 *
		 * @type {Array<Node>}
		 */
		this.resources = [];

	}

	setup( builder ) {

		this.resources = this.plan( builder );

		return super.setup( builder ); // registers the resources as children

	}

	generate( builder ) {

		for ( const node of this.resources ) node.build( builder, 'property' );

		return '';

	}

}

export { RawResourcesNode };
